import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D convolution
conv2d_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename scalar_t>
__global__ void custom_conv2d_forward(const scalar_t* __restrict__ input,
                                     const scalar_t* __restrict__ weight,
                                     scalar_t* output,
                                     const int batch_size,
                                     const int in_channels,
                                     const int out_channels,
                                     const int kernel_size,
                                     const int height,
                                     const int width,
                                     const int stride,
                                     const int padding,
                                     const int dilation) {
    const int output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z / out_channels;
    const int out_ch = blockIdx.z % out_channels;

    if (col >= output_width || row >= output_height || b >= batch_size) return;

    scalar_t sum = 0;
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int ki = 0; ki < kernel_size; ++ki) {
            for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                const int h_in = -padding + row * stride + kd * dilation;
                const int w_in = -padding + col * stride + ki * dilation;

                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    sum += input[b * in_channels * height * width + in_ch * height * width + h_in * width + w_in] *
                           weight[out_ch * in_channels * kernel_size * kernel_size + in_ch * kernel_size * kernel_size + kd * kernel_size + ki];
                }
            }
        }
    }
    output[b * out_channels * output_height * output_width + out_ch * output_height * output_width + row * output_width + col] = sum;
}

torch::Tensor custom_conv2d_forward_cuda(torch::Tensor input,
                                        torch::Tensor weight,
                                        int stride,
                                        int padding,
                                        int dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int height = input.size(2);
    const int width = input.size(3);

    const int output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 threads(16, 16, 1);
    dim3 blocks(output_width / threads.x + 1,
                output_height / threads.y + 1,
                batch_size * out_channels);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv2d_forward_cuda", ([&] {
        custom_conv2d_forward<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            height,
            width,
            stride,
            padding,
            dilation);
    }));

    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor custom_conv2d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"
)

# Compile the CUDA kernel
custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_kernel_source,
    functions=["custom_conv2d_forward_cuda"],
    verbose=True,
    extra_cflags=["-DGLenum=void"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        torch.nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        output = custom_conv.custom_conv2d_forward_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output