import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

# Define custom CUDA kernels for optimized 3D convolution
current_dir = os.path.dirname(os.path.abspath(__file__))
conv3d_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void optimized_conv3d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,5> input,
    const torch::PackedTensorAccessor<scalar_t,5> weight,
    torch::PackedTensorAccessor<scalar_t,5> output,
    int batch_size, int in_channels, int out_channels,
    int kernel_width, int kernel_height, int kernel_depth,
    int stride, int padding, int dilation) {

    const int output_depth = output.size(4);
    const int output_height = output.size(3);
    const int output_width = output.size(2);
    const int output_channels = output.size(1);

    const int depth_radius = (kernel_depth - 1) * dilation + 1;
    const int height_radius = (kernel_height - 1) * dilation + 1;
    const int width_radius = (kernel_width - 1) * dilation + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * output_channels * output_width * output_height * output_depth) {
        return;
    }

    int d = idx % output_depth;
    idx /= output_depth;
    int h = idx % output_height;
    idx /= output_height;
    int w = idx % output_width;
    idx /= output_width;
    int c = idx % output_channels;
    int n = idx / output_channels;

    scalar_t sum = 0;
    for (int kd = 0; kd < kernel_depth; ++kd) {
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int in_d = d * stride - padding + kd * dilation;
                int in_h = h * stride - padding + kh * dilation;
                int in_w = w * stride - padding + kw * dilation;
                if (in_d < 0 || in_d >= input.size(4) ||
                    in_h < 0 || in_h >= input.size(3) ||
                    in_w < 0 || in_w >= input.size(2)) {
                    continue;
                }
                for (int ic = 0; ic < in_channels; ++ic) {
                    sum += input[n][ic][in_w][in_h][in_d] *
                           weight[c][ic][kw][kh][kd];
                }
            }
        }
    }
    output[n][c][w][h][d] = sum;
}

at::Tensor optimized_conv3d_forward_cuda(
    at::Tensor input,
    at::Tensor weight,
    int stride,
    int padding,
    int dilation) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_width = weight.size(2);
    const int kernel_height = weight.size(3);
    const int kernel_depth = weight.size(4);

    int output_width = (input.size(2) + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;
    int output_height = (input.size(3) + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int output_depth = (input.size(4) + 2 * padding - dilation * (kernel_depth - 1) - 1) / stride + 1;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    at::Tensor output = torch::zeros({batch_size, out_channels, output_width, output_height, output_depth}, output_options);

    int total_threads = batch_size * output_channels * output_width * output_height * output_depth;
    int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "optimized_conv3d_forward_cuda", ([&] {
        optimized_conv3d_forward_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.packed_accessor<scalar_t,5>(),
            weight.packed_accessor<scalar_t,5>(),
            output.packed_accessor<scalar_t,5>(),
            batch_size, in_channels, out_channels,
            kernel_width, kernel_height, kernel_depth,
            stride, padding, dilation);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

# Compile the CUDA code
conv3d_cuda = load(
    name="optimized_conv3d",
    sources=[conv3d_kernel_source],
    extra_cuda_cflags=['-gencode=arch=compute_80,code=sm_80'], # For Ampere architecture
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv3d_cuda.optimized_conv3d_forward_cuda(
            x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output