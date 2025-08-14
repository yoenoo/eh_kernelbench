import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import numpy as np

# Custom CUDA kernel code
conv3d_kernel = '''
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void custom_conv3d_kernel(const scalar_t* __restrict__ input,
                                    const scalar_t* __restrict__ weight,
                                    scalar_t* __restrict__ output,
                                    int64_t batch_size,
                                    int64_t in_channels,
                                      int64_t in_depth,
                                      int64_t in_height,
                                      int64_t in_width,
                                      int64_t out_channels,
                                      int64_t kernel_d,
                                      int64_t kernel_h,
                                      int64_t kernel_w,
                                      int64_t stride_d,
                                      int64_t stride_h,
                                      int64_t stride_w,
                                      int64_t pad_d,
                                      int64_t pad_h,
                                      int64_t pad_w,
                                      int64_t dilation_d,
                                      int64_t dilation_h,
                                      int64_t dilation_w,
                                      int64_t out_depth,
                                      int64_t out_height,
                                      int64_t out_width) {

    const int channels_per_block = 32;
    const int output_channels = out_channels;
    const int blocks_needed = (output_channels + channels_per_block - 1) / channels_per_block;

    int channel_block = blockIdx.z % blocks_needed;
    int output_depth = blockIdx.x * blockDim.z + threadIdx.z;
    int output_height = blockIdx.y * blockDim.y + threadIdx.y;
    int output_width = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_depth >= out_depth || output_height >= out_height || output_width >= out_width) return;

    for (int c = channel_block * channels_per_block; c < output_channels && c < (channel_block + 1)*channels_per_block; c += 1) {
        scalar_t sum = 0;
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int input_d = output_depth * stride_d - pad_d + kd * dilation_d;
                    int input_h = output_height * stride_h - pad_h + kh * dilation_h;
                    int input_w = output_width * stride_w - pad_w + kw * dilation_w;
                    if (input_d >= 0 && input_d < in_depth && input_h >=0 && input_h < in_height && input_w >=0 && input_w < in_width) {
                        for (int ic = 0; ic < in_channels; ++ic) {
                            int w_idx = c * in_channels * kernel_d * kernel_h * kernel_w + ic * kernel_d * kernel_h * kernel_w +
                                        kd * kernel_h * kernel_w + kh * kernel_w + kw;
                            int in_idx = ic + in_channels * (input_d + in_depth * (input_h + in_height * input_w));
                            sum += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
        int out_idx = c + output_channels * (output_depth + out_depth * (output_height + out_height * output_width));
        output[out_idx] = sum;
    }
}

at::Tensor custom_conv3d(at::Tensor input, at::Tensor weight,
                        int stride_d, int stride_h, int stride_w,
                        int pad_d, int pad_h, int pad_w,
                        int dilation_d, int dilation_h, int dilation_w) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);

    const int out_depth = (in_depth + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    const int out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    at::Tensor output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    int threads = 256;
    int blocks_x = (out_width + threads - 1) / threads;
    int blocks_y = (out_height + threads - 1) / threads;
    int blocks_z = (out_depth + threads - 1) / threads;

    int blocks_needed = ((out_channels + 31) / 32); // 32 channels per block
    dim3 block_dim(threads, threads, threads);
    dim3 grid_dim(blocks_x, blocks_y, blocks_z * blocks_needed);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv3d", ([&] {
        custom_conv3d_kernel<scalar_t><<<grid_dim, block_dim>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, in_depth, in_height, in_width,
            out_channels, kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            dilation_d, dilation_h, dilation_w,
            out_depth, out_height, out_width);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_conv3d", &custom_conv3d, "Custom 3D convolution kernel");
}
'''

# Compile the extension
conv3d_cuda = load(name="custom_conv3d", sources=[conv3d_kernel], verbose=True)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False):
        super().__init__()
        # Initialize convolution weights and parameters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Weight initialization
        weight = torch.randn(out_channels, in_channels // groups, *kernel_size)
        self.weight = nn.Parameter(weight)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        self.groups = groups

    def forward(self, x):
        # Call the custom CUDA kernel
        output = conv3d_cuda.custom_conv3d(
            x, self.weight,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.dilation[0], self.dilation[1], self.dilation[2]
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        return output