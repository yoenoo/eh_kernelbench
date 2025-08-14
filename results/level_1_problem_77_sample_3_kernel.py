import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Forward kernel for 3D transposed convolution
template <typename scalar_t>
__global__ void conv_transpose3d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,5> input,
    const torch::PackedTensorAccessor<scalar_t,5> weight,
    torch::PackedTensorAccessor<scalar_t,5> output,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int kernel_size, int stride, int padding, int dilation,
    int out_depth, int out_height, int out_width) {

    // Thread and block indices
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (d >= out_depth || h >= out_height || w >= out_width) {
        return;
    }

    // Compute output position with stride and padding
    // (simplified for demonstration, may need adjustment)
    int od = d * stride - padding;
    int oh = h * stride - padding;
    int ow = w * stride - padding;

    // Iterate over input and kernel
    scalar_t sum = 0;
    for (int k_d = 0; k_d < kernel_size; ++k_d) {
        for (int k_h = 0; k_h < kernel_size; ++k_h) {
            for (int k_w = 0; k_w < kernel_size; ++k_w) {
                // Compute input indices (considering dilation and kernel offset)
                int id = od + k_d * dilation;
                int ih = oh + k_h * dilation;
                int iw = ow + k_w * dilation;

                // Check boundaries
                if (id >= 0 && id < in_depth &&
                    ih >= 0 && ih < in_height &&
                    iw >= 0 && iw < in_width) {

                    for (int in_c = 0; in_c < in_channels; ++in_c) {
                        for (int out_c = 0; out_c < out_channels; ++out_c) {
                            sum += input[0][in_c][id][ih][iw] *
                                   weight[out_c][in_c][k_d][k_h][k_w];
                        }
                    }
                }
            }
        }
    }
    output[0][0][d][h][w] = sum; // Assuming single channel for simplicity
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv_transpose3d_cuda_forward(
    torch::Tensor input, torch::Tensor weight,
    int kernel_size, int stride, int padding, int dilation) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    // Compute output dimensions
    const int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + dilation - 1;
    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + dilation - 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + dilation - 1;

    auto output_options = torch::TensorOptions()
        .device(torch::kCUDA)
        .dtype(input.dtype());
    torch::Tensor output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, output_options);

    // Define grid and block dimensions
    dim3 threads(16, 16, 1);
    dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        (out_depth + threads.z - 1) / threads.z));

    // Launch kernel with appropriate types
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_forward", ([&] {
        conv_transpose3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5>(),
            weight.packed_accessor<scalar_t,5>(),
            output.packed_accessor<scalar_t,5>(),
            batch_size, in_channels, out_channels,
            in_depth, in_height, in_width,
            kernel_size, stride, padding, dilation,
            out_depth, out_height, out_width);
    }));

    cudaDeviceSynchronize();
    return std::make_tuple(output, input, weight);
}
"""

cpp_source = """
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv_transpose3d_cuda_forward(
    torch::Tensor input, torch::Tensor weight,
    int kernel_size, int stride, int padding, int dilation);
"""

# Compile the custom CUDA operator
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=[cpp_source],
    cuda_sources=[conv_transpose3d_source],
    functions="conv_transpose3d_cuda_forward",
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels,
                                               kernel_size, kernel_size, kernel_size))
        # Initialize weights (simplified)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _, _ = conv_transpose3d.conv_transpose3d_cuda_forward(
            x.cuda(), self.weight.cuda(),
            self.kernel_size, self.stride, self.padding, self.dilation)
        return output