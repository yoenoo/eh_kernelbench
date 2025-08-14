import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv2d_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename scalar_t>
__global__ void custom_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int input_h,
    const int input_w,
    const int output_h,
    const int output_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w
) {
    const int output_size = batch_size * out_channels * output_h * output_w;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    const int w = idx % output_w;
    const int h = (idx / output_w) % output_h;
    const int c_out = (idx / output_w / output_h) % out_channels;
    const int n = idx / output_w / output_h / out_channels;

    scalar_t val = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int h_in = -pad_h + h * stride_h + kh * dilation_h;
            const int w_in = -pad_w + w * stride_w + kw * dilation_w;
            if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    val += weight[c_out * in_channels * kernel_h * kernel_w + c_in * kernel_h * kernel_w + kh * kernel_w + kw] *
                           input[n * in_channels * input_h * input_w + c_in * input_h * input_w + h_in * input_w + w_in];
                }
            }
        }
    }
    output[idx] = val;
}

torch::Tensor custom_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels = weight.size(0);

    // Compute output dimensions using conv formula
    const int output_h = (input_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int output_w = (input_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    int threads = 256;
    int blocks = (batch_size * out_channels * output_h * output_w + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv2d_cuda", ([&] {
        custom_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            input_h,
            input_w,
            output_h,
            output_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv2d_kernel_cpp = "torch::Tensor custom_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w);"

custom_conv2d = load_inline(
    name="custom_conv2d",
    cpp_sources=[conv2d_kernel_cpp],
    cuda_sources=[conv2d_kernel_source],
    functions=["custom_conv2d_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Initialize convolution weights manually
        kernel_h, kernel_w = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_h, kernel_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        # Custom convolution without bias for simplicity (bias can be added later)
        output = custom_conv2d.custom_conv2d_cuda(
            x,
            self.weight,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1]
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output