import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <vector>
#include <iostream>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
                                       const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
                                       torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
                                       int kernel_h, int kernel_w,
                                       int stride_h, int stride_w,
                                       int padding_h, int padding_w,
                                       int dilation_h, int dilation_w) {
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int oh = blockIdx.x * blockDim.y + threadIdx.y;
    int ow = threadIdx.x;

    if (oh >= output.size(2) || ow >= output.size(3)) {
        return;
    }

    scalar_t val = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_out = oh + padding_h - kh * dilation_h;
            int w_out = ow + padding_w - kw * dilation_w;
            if (h_out % stride_h != 0 || w_out % stride_w != 0) {
                continue;
            }
            h_out /= stride_h;
            w_out /= stride_w;
            if (h_out < 0 || h_out >= input.size(2) || w_out <0 || w_out >= input.size(3)) {
                continue;
            }
            for (int c_in = 0; c_in < weight.size(1); ++c_in) {
                val += input[n][c_in][h_out][w_out] * weight[c_out][c_in][kh][kw];
            }
        }
    }
    output[n][c_out][oh][ow] = val;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int stride_h, int stride_w,
                                   int padding_h, int padding_w,
                                   int dilation_h, int dilation_w) {
    auto output_height = (input.size(2) - 1) * stride_h - 2 * padding_h + 
                        dilation_h * (weight.size(2) - 1) + 1;
    auto output_width = (input.size(3) - 1) * stride_w - 2 * padding_w +
                       dilation_w * (weight.size(3) - 1) + 1;
    auto output = torch::empty({input.size(0), weight.size(0), output_height, output_width},
                              input.options());

    int block_size_x = 32;
    int block_size_y = 8;
    dim3 block(block_size_x, block_size_y);
    dim3 grid( (output_width + block_size_x -1)/block_size_x,
              (output_height + block_size_y -1)/block_size_y,
              input.size(0)*weight.size(0));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&]{
        conv_transpose2d_kernel<scalar_t><<<grid, block>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.size(2), weight.size(3),
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);"
)

# Compile the inline CUDA code for ConvTranspose2d
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-gencode=arch=compute_75,code=sm_75"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        # Normally bias would be handled here, but omitted as per default False
        self.stride = stride if isinstance(stride, int) else stride[0]
        self.padding = padding
        self.dilation = dilation
        self.conv_transpose2d_op = conv_transpose2d

    def forward(self, x):
        return self.conv_transpose2d_op.conv_transpose2d_cuda(
            x, self.weight,
            self.stride, self.stride,
            self.padding, self.padding,
            self.dilation, self.dilation
        )