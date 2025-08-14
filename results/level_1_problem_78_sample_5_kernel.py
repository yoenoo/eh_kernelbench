import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

# Define the custom CUDA kernel source code for ConvTranspose2d
conv_transpose2d_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// This is a simplified kernel and may not handle all edge cases
template <typename scalar_t>
__global__ void conv_transpose2d_kernel(const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ weight,
                                       scalar_t* __restrict__ output,
                                       const int batch_size,
                                       const int in_channels,
                                       const int out_channels,
                                       const int kernel_h,
                                       const int kernel_w,
                                       const int input_h,
                                       const int input_w,
                                       const int stride_h,
                                       const int stride_w,
                                       const int padding_h,
                                       const int padding_w,
                                       const int output_h,
                                       const int output_w) {

    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (output_idx >= batch_size * out_channels * output_h * output_w) return;

    int w = output_idx % output_w;
    int h = (output_idx / output_w) % output_h;
    int c_out = (output_idx / (output_w * output_h)) % out_channels;
    int n = output_idx / (out_channels * output_w * output_h);

    scalar_t val = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            // Compute the corresponding input position
            int input_h_idx = (h - padding_h - kh) / stride_h;
            int input_w_idx = (w - padding_w - kw) / stride_w;
            if (input_h_idx < 0 || input_h_idx >= input_h ||
                input_w_idx < 0 || input_w_idx >= input_w) {
                continue;
            }
            for (int c_in = 0; c_in < in_channels; ++c_in) {
                // Weight indices: weight is [in_channels, out_channels, kernel_h, kernel_w]
                int weight_offset = c_in * out_channels * kernel_h * kernel_w +
                                    c_out * kernel_h * kernel_w +
                                    kh * kernel_w + kw;
                // Input indices: [n, c_in, input_h, input_w]
                int input_offset = n * in_channels * input_h * input_w +
                                   c_in * input_h * input_w +
                                   input_h_idx * input_w + input_w_idx;
                val += input[input_offset] * weight[weight_offset];
            }
        }
    }
    output[output_idx] = val;
}

// Interface for the kernel
at::Tensor conv_transpose2d_cuda(at::Tensor input, at::Tensor weight,
                                 int stride_h, int stride_w,
                                 int padding_h, int padding_w,
                                 int output_padding_h, int output_padding_w) {
    // Calculate output dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_h = input.size(2);
    auto input_w = input.size(3);
    auto out_channels = weight.size(1);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);

    int output_h = (input_h - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int output_w = (input_w - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = at::empty({batch_size, out_channels, output_h, output_w}, input.options());

    const int threads = 1024;
    int elements = output.numel();
    int blocks = (elements + threads - 1) / threads;

    // Launch the kernel with appropriate template type
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, out_channels,
            kernel_h, kernel_w, input_h, input_w,
            stride_h, stride_w, padding_h, padding_w,
            output_h, output_w);
    }));

    return output;
}
"""

# Additional headers and sources for compilation
cpp_source = """
#include <torch/extension.h>
"""

# Load the CUDA extension
module = load(name='conv_transpose2d_kernel',
              sources=[conv_transpose2d_kernel, cpp_source],
              extra_cflags=['-std=c++14'],
              verbose=True)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = (0, 0)  # Assuming no output padding for simplicity

        # Initialize weights similar to PyTorch's ConvTranspose2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) 

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x):
        output = module.conv_transpose2d_cuda(x, self.weight, 
                                              self.stride[0], self.stride[1],
                                              self.padding[0], self.padding[1],
                                              self.output_padding[0], self.output_padding[1])
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output