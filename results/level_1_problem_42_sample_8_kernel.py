import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for Max Pooling 2D
maxpool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input,
                                          torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output,
                                          int kernel_size,
                                          int stride,
                                          int padding,
                                          int dilation,
                                          int batch_size,
                                          int channels,
                                          int input_height,
                                          int input_width,
                                          int output_height,
                                          int output_width) {
    const int batch_idx = blockIdx.x;
    const int channel_idx = blockIdx.y;
    const int output_y = threadIdx.y;
    const int output_x = threadIdx.x;

    // Compute input coordinates with padding
    const int in_y_start = -padding + output_y * stride;
    const int in_x_start = -padding + output_x * stride;

    scalar_t max_val = -FLT_MAX;
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            const int ry = in_y_start + ky * dilation;
            const int rx = in_x_start + kx * dilation;
            if (ry >= 0 && ry < input_height && rx >= 0 && rx < input_width) {
                const scalar_t val = input[batch_idx][channel_idx][ry][rx];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    output[batch_idx][channel_idx][output_y][output_x] = max_val;
}

std::tuple<torch::Tensor> max_pool2d_forward_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    const auto& input_size = input.size();
    const int batch_size = input_size[0];
    const int channels = input_size[1];
    const int input_height = input_size[2];
    const int input_width = input_size[3];

    // Compute output dimensions
    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    torch::Tensor output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    dim3 threads(output_width, output_height);
    dim3 blocks(batch_size, channels);

    const int block_size_x = output_width;
    const int block_size_y = output_height;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward_cuda", ([&] {
        max_pool2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            kernel_size,
            stride,
            padding,
            dilation,
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

maxpool2d_cpp_source = (
    "std::tuple<torch::Tensor> max_pool2d_forward_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);"
)

# Compile the inline CUDA code for max pooling
max_pool2d = load_inline(
    name="max_pool2d",
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_source,
    functions=["max_pool2d_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return max_pool2d.max_pool2d_forward_cuda(
            x, self.kernel_size, self.stride, self.padding, self.dilation
        )[0]

batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]