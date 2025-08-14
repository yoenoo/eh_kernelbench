import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 3D convolution
transposed_conv3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename scalar_t>
__global__ void transposed_conv3d_kernel(const scalar_t* input,
                                        const scalar_t* weight,
                                        scalar_t* output,
                                        int batch_size,
                                        int in_channels,
                                        int out_channels,
                                        int kernel_d,
                                        int kernel_h,
                                        int kernel_w,
                                        int stride_d,
                                        int stride_h,
                                        int stride_w,
                                        int padding_d,
                                        int padding_h,
                                        int padding_w,
                                        int output_padding_d,
                                        int output_padding_h,
                                        int output_padding_w,
                                        int depth_in,
                                        int height_in,
                                        int width_in,
                                        int depth_out,
                                        int height_out,
                                        int width_out) {

    int batch_idx = blockIdx.x;
    int out_z = threadIdx.z;
    int out_y = threadIdx.y;
    int out_x = threadIdx.x;

    for (int out_d = blockIdx.z; out_d < depth_out; out_d += gridDim.z) {
        for (int out_h = blockIdx.y; out_h < height_out; out_h += gridDim.y) {
            for (int out_w = blockIdx.x; out_w < width_out; out_w += gridDim.x) {
                // Compute the input coordinates based on transposed logic
                int in_d = (out_d - output_padding_d) / stride_d - padding_d;
                int in_h = (out_h - output_padding_h) / stride_h - padding_h;
                int in_w = (out_w - output_padding_w) / stride_w - padding_w;

                if (in_d < 0 || in_h < 0 || in_w < 0) continue;

                // Iterate over kernel elements
                for (int kd = 0; kd < kernel_d; ++kd) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int k_offset = kd * kernel_h * kernel_w + kh * kernel_w + kw;
                            for (int c = 0; c < in_channels; ++c) {
                                int in_offset = c * depth_in * height_in * width_in 
                                                + (in_d + kd) * height_in * width_in 
                                                + (in_h + kh) * width_in 
                                                + (in_w + kw);
                                int out_offset = batch_idx * out_channels * depth_out * height_out * width_out 
                                                + out_channels * (out_d * height_out * width_out 
                                                                  + out_h * width_out + out_w) 
                                                + c * kernel_d * kernel_h * kernel_w + k_offset;
                                atomicAdd(&output[out_offset], 
                                          input[in_offset] * weight[out_offset]);
                            }
                        }
                    }
                }
            }
        }
    }
}

at::Tensor transposed_conv3d_cuda(at::Tensor input, at::Tensor weight,
                                 int kernel_d, int kernel_h, int kernel_w,
                                 int stride_d, int stride_h, int stride_w,
                                 int padding_d, int padding_h, int padding_w,
                                 int output_padding_d, int output_padding_h,
                                 int output_padding_w) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int depth_in = input.size(2);
    const int height_in = input.size(3);
    const int width_in = input.size(4);
    const int depth_out = (depth_in - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    const int height_out = (height_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    const int width_out = (width_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output_opts = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, output_opts);

    dim3 threads(16, 16, 4); // Thread block dimensions
    dim3 blocks(ceil(width_out / 16.0), ceil(height_out / 16.0), ceil(depth_out / 4.0));

    transposed_conv3d_kernel<float><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size, in_channels, out_channels,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        depth_in, height_in, width_in,
        depth_out, height_out, width_out);

    return output;
}
"""

transposed_conv3d_cpp_source = """
at::Tensor transposed_conv3d_cuda(at::Tensor input, at::Tensor weight,
                                 int kernel_d, int kernel_h, int kernel_w,
                                 int stride_d, int stride_h, int stride_w,
                                 int padding_d, int padding_h, int padding_w,
                                 int output_padding_d, int output_padding_h, int output_padding_w);
"""

transposed_conv3d = load_inline(
    name='transposed_conv3d',
    cpp_sources=transposed_conv3d_cpp_source,
    cuda_sources=transposed_conv3d_source,
    functions=['transposed_conv3d_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=[''],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 output_padding: tuple = (0, 0, 0), groups: int = 1,
                 bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) 

    def forward(self, x):
        return transposed_conv3d.transposed_conv3d_cuda(
            x, self.weight,
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2]
        )