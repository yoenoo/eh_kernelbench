import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose2D
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(scalar_t *input, scalar_t *weight, scalar_t *output,
                          int batch_size, int in_channels, int out_channels,
                          int kernel_h, int kernel_w, int stride_h, int stride_w,
                          int output_padding_h, int output_padding_w,
                          int input_h, int input_w,
                          int output_h, int output_w,
                          int padding_h, int padding_w,
                          int dilation_h, int dilation_w) {

    // Implement the forward pass of the transposed convolution here
    // Index calculation for the output tensor
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx = blockIdx.z;

    if (b_idx >= batch_size || h_idx >= output_h || w_idx >= output_w) {
        return;
    }

    // Iterate over the input channels and kernel dimensions
    for (int c_out = 0; c_out < out_channels; ++c_out) {
        scalar_t val = 0;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    // Compute the corresponding input position
                    int h_in = (h_idx - kh * stride_h - output_padding_h) / stride_h;
                    int w_in = (w_idx - kw * stride_w - output_padding_w) / stride_w;

                    // Check if the current position is within input bounds and valid
                    if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                        // Access input and weight
                        val += input[b_idx * in_channels * input_h * input_w + c_in * input_h * input_w + h_in * input_w + w_in] *
                               weight[c_out * in_channels * kernel_h * kernel_w + c_in * kernel_h * kernel_w + kh * kernel_w + kw];
                    }
                }
            }
        }
        // Write the result to the output
        output[b_idx * out_channels * output_h * output_w + c_out * output_h * output_w + h_idx * output_w + w_idx] = val;
    }
}

at::Tensor conv_transpose2d_cuda(at::Tensor input, at::Tensor weight,
                              int stride_h, int stride_w,
                              int padding_h, int padding_w,
                              int output_padding_h, int output_padding_w) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    const int output_h = (input_h - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    const int output_w = (input_w - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    at::Tensor output = at::empty({batch_size, out_channels, output_h, output_w}, input.options());

    dim3 threads(32, 8);
    dim3 blocks(batch_size, (output_h + threads.y - 1) / threads.y, (output_w + threads.x - 1) / threads.x);

    conv_transpose2d_kernel<float><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size, in_channels, out_channels,
        kernel_h, kernel_w, stride_h, stride_w,
        output_padding_h, output_padding_w,
        input_h, input_w,
        output_h, output_w,
        padding_h, padding_w,
        1, 1);

    return output;
}
"""

conv_transpose2d_cpp_source = """
at::Tensor conv_transpose2d_cuda(at::Tensor input, at::Tensor weight,
                              int stride_h, int stride_w,
                              int padding_h, int padding_w,
                              int output_padding_h, int output_padding_w);
"""

# Compile the custom CUDA kernel
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=[conv_transpose2d_cpp_source],
    cuda_sources=[conv_transpose2d_source],
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.kernel_size = (kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assuming 2D kernel (height == width)
        output = conv_transpose2d.conv_transpose2d_cuda(x, self.weight, self.stride, self.stride,
                                                        self.padding, self.padding,
                                                        self.output_padding, self.output_padding)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output