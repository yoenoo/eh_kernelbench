import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(scalar_t* input, scalar_t* weight, scalar_t* output,
    int batch_size, int in_channels, int out_channels, int kH, int kW,
    int input_H, int input_W, int output_H, int output_W,
    int stride_H, int stride_W, int pad_H, int pad_W) {

    int B = blockIdx.z;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_y >= output_H || out_x >= output_W) return;

    for (int c_out = threadIdx.z; c_out < out_channels; c_out += blockDim.z) {
        scalar_t sum = 0;
        for (int ky = 0; ky < kH; ++ky) {
            for (int kx = 0; kx < kW; ++kx) {
                int in_y = out_y * stride_H - pad_H + ky;
                int in_x = out_x * stride_W - pad_W + kx;
                if (in_y < 0 || in_y >= input_H || in_x < 0 || in_x >= input_W) continue;
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    sum += input[B * in_channels * input_H * input_W +
                                c_in * input_H * input_W +
                                in_y * input_W + in_x] *
                            weight[c_out * in_channels * kH * kW +
                                c_in * kH * kW +
                                ky * kW + kx];
                }
            }
        }
        output[B * out_channels * output_H * output_W +
            c_out * output_H * output_W +
            out_y * output_W + out_x] = sum;
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight,
    int stride_H, int stride_W, int pad_H, int pad_W) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kH = weight.size(2);
    const int kW = weight.size(3);
    const int input_H = input.size(2);
    const int input_W = input.size(3);
    const int output_H = (input_H - 1) * stride_H - 2 * pad_H + kH;
    const int output_W = (input_W - 1) * stride_W - 2 * pad_W + kW;

    auto output = torch::zeros({batch_size, out_channels, output_H, output_W}, input.options());

    const int block_size_x = 32;
    const int block_size_y = 8;
    const int block_size_z = 16;

    dim3 block(block_size_x, block_size_y, block_size_z);
    dim3 grid(
        (output_W + block_size_x - 1) / block_size_x,
        (output_H + block_size_y - 1) / block_size_y,
        batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<grid, block>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, out_channels,
            kH, kW, input_H, input_W, output_H, output_W,
            stride_H, stride_W, pad_H, pad_W);
        }));
    return output;
}
"""

conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_H, int stride_W, int pad_H, int pad_W);"
)

# Compile the custom CUDA kernel
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_API__"],
    extra_cuda_cflags=["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_API__"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        # Initialize weight like PyTorch's ConvTranspose2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run custom CUDA kernel
        out = conv_transpose2d.conv_transpose2d_cuda(
            x.cuda(),
            self.weight.cuda(),
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
        )
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out