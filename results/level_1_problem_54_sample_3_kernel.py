import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom 3D Convolution CUDA Kernel
conv3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the 3D convolution kernel
template <typename scalar_t>
__global__ void conv3d_kernel(const torch::PackedTensorAccessor<scalar_t,5> input,
                             const torch::PackedTensorAccessor<scalar_t,5> weight,
                             torch::PackedTensorAccessor<scalar_t,5> output,
                             const int batch_size, const int in_channels,
                             const int in_depth, const int in_height, const int in_width,
                             const int out_channels, const int kernel_size,
                             const int stride, const int padding, const int dilation) {
    // Thread and block indices
    const int b = blockIdx.x;
    const int oz = threadIdx.z;
    const int oh = threadIdx.y;
    const int ow = threadIdx.x;

    // Output dimensions
    const int out_depth = (in_depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Each thread computes one output element
    for (int c = 0; c < out_channels; c += blockDim.z) {
        const int out_z = oz + c;
        if (out_z >= out_channels) continue;

        scalar_t sum = 0;
        for (int iz = 0; iz < kernel_size; ++iz) {
            for (int ih = 0; ih < kernel_size; ++ih) {
                for (int iw = 0; iw < kernel_size; ++iw) {
                    // Compute input positions
                    const int in_z = oz * stride + dilation * iz - padding;
                    const int in_h = oh * stride + dilation * ih - padding;
                    const int in_w = ow * stride + dilation * iw - padding;

                    // Boundary check
                    if (in_z < 0 || in_z >= in_depth || in_h < 0 || in_h >= in_height || in_w < 0 || in_w >= in_width) {
                        continue;
                    }

                    for (int ic = 0; ic < in_channels; ++ic) {
                        sum += input[b][ic][in_z][in_h][in_w] * weight[c][ic][iz][ih][iw];
                    }
                }
            }
        }
        output[b][out_z][oh][ow][c] = sum;
    }
}

// C++ entry point
at::Tensor conv3d_cuda(const at::Tensor &input, const at::Tensor &weight,
                      const int stride, const int padding, const int dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    // Calculate output dimensions
    const int out_depth = (in_depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Output tensor
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    // Block and grid dimensions
    dim3 threads(16, 16, 16);
    dim3 blocks(batch_size, 1, 1);

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_cuda", ([&] {
        conv3d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5>(),
            weight.packed_accessor<scalar_t,5>(),
            output.packed_accessor<scalar_t,5>(),
            batch_size, in_channels, in_depth, in_height, in_width,
            out_channels, kernel_size, stride, padding, dilation
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv3d_cpp_source = """
at::Tensor conv3d_cuda(const at::Tensor &input, const at::Tensor &weight,
                      const int stride, const int padding, const int dilation);
"""

# Compile the CUDA kernel
conv3d = load_inline(
    name="conv3d_cuda",
    cpp_sources=[conv3d_cpp_source],
    cuda_sources=[conv3d_source],
    functions=["conv3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 
                                               kernel_size, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Xavier initialization
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        out = conv3d.conv3d_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1, 1)
        return out