import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>

// Define the CUDA kernel for ConvTranspose2d.
template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int in_channels, int out_channels, int kernel_size, int stride, int padding, int output_padding) {

    // Implementation details go here (this is an example skeleton)
    const int batch_size = input.size(0);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_height = output.size(2);
    const int out_width = output.size(3);

    const int channels_per_group = in_channels / groups;

    const int output_batch = blockIdx.x;
    const int output_channel = blockIdx.y;
    const int output_y = blockIdx.z * blockDim.y + threadIdx.y;
    const int output_x = threadIdx.x;

    if (output_y >= out_height || output_x >= out_width) {
        return;
    }

    scalar_t sum = 0;
    for (int input_channel = 0; input_channel < channels_per_group; input_channel++) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                // Calculate input coordinates
                int input_y = (output_y - ky) / stride;
                int input_x = (output_x - kx) / stride;
                
                // Check bounds and contribute to sum
                if ((output_y - ky) % stride == 0 && input_y >= 0 && input_y < in_height &&
                    (output_x - kx) % stride == 0 && input_x >= 0 && input_x < in_width) {
                    sum += weight[output_channel][input_channel][ky][kx] *
                            input[output_batch][input_channel][input_y][input_x];
                }
            }
        }
    }
    output[output_batch][output_channel][output_y][output_x] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input, torch::Tensor weight,
    int stride, int padding, int output_padding, int groups) {

    // Get necessary dimensions and prepare output tensor
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int kernel_size = weight.size(2); // Assuming square kernel
    const int out_channels = weight.size(0);

    // Compute output shape (this is a simplified example; actual calculation requires proper formulas)
    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({batch_size, out_channels, out_height, out_width}, output_options);

    int threads = 256;
    dim3 blocks(batch_size, out_channels, (out_height * out_width + threads - 1) / threads);
    dim3 threadsPerBlock(threads, 1);

    // Launch the kernel (type promotion may be needed)
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "conv_transpose2d_cuda", ([&]{
        conv_transpose2d_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            in_channels, out_channels, kernel_size, stride, padding, output_padding);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input, torch::Tensor weight,
    int stride, int padding, int output_padding, int groups
);
"""

conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weights similar to PyTorch's ConvTranspose2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return conv_transpose.conv_transpose2d_cuda(
            x, self.weight, self.stride, self.padding, self.output_padding, self.groups
        )