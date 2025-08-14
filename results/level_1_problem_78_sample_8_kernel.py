import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA implementation for ConvTranspose2d
conv_transpose_2d_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(const scalar_t* bottom_data, scalar_t* top_data,
        const scalar_t* weights, const int num_images, const int input_channels,
        const int input_height, const int input_width,
        const int output_channels, const int kernel_h, const int kernel_w,
        const int stride_h, const int stride_w, const int pad_h, const int pad_w,
        const int output_height, const int output_width) {

    CUDA_KERNEL_LOOP(index, num_images * input_channels * input_height * input_width) {
        int w = index % input_width;
        int h = (index / input_width) % input_height;
        int c = (index / input_width / input_height) % input_channels;
        int n = index / input_width / input_height / input_channels;

        const scalar_t* offset_bottom = bottom_data + (n * input_channels + c) * input_height * input_width + h * input_width + w;

        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_out = h * stride_h - pad_h + kh;
                int w_out = w * stride_w - pad_w + kw;
                if (h_out < 0 || h_out >= output_height || w_out < 0 || w_out >= output_width) {
                    continue;
                }
                for (int c_out = 0; c_out < output_channels; ++c_out) {
                    const int weight_index = c_out * input_channels * kernel_h * kernel_w +
                                            c * kernel_h * kernel_w + kh * kernel_w + kw;
                    const scalar_t weight = weights[weight_index];
                    const int top_index = n * output_channels * output_height * output_width +
                        c_out * output_height * output_width + h_out * output_width + w_out;
                    atomicAdd(top_data + top_index, *offset_bottom * weight);
                }
            }
        }
    }
}

torch::Tensor conv_transpose2d_forward(torch::Tensor input, torch::Tensor weight,
        int kernel_h, int kernel_w,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    const int num_images = input.size(0);
    const int input_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int output_channels = weight.size(0);
    const int output_height = (input_height - 1) * stride_h - 2 * pad_h + kernel_h;
    const int output_width = (input_width - 1) * stride_w - 2 * pad_w + kernel_w;

    auto output = torch::zeros({num_images, output_channels, output_height, output_width}, input.options());

    const int total_threads = num_images * input_channels * input_height * input_width;

    const int block_size = 256;
    const int grid_size = (total_threads + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_forward", ([&] {
        conv_transpose2d_kernel<scalar_t><<<grid_size, block_size>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            weight.data<scalar_t>(),
            num_images, input_channels,
            input_height, input_width,
            output_channels, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w,
            output_height, output_width);
    }));

    return output;
}
"""

conv_transpose_2d_cpp = """
torch::Tensor conv_transpose2d_forward(torch::Tensor input, torch::Tensor weight,
        int kernel_h, int kernel_w,
        int stride_h, int stride_w,
        int pad_h, int pad_w);
"""

conv_transpose_module = load_inline(
    name="conv_transpose_module",
    cpp_sources=conv_transpose_2d_cpp,
    cuda_sources=conv_transpose_2d_source,
    functions="conv_transpose2d_forward",
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        # Bias is omitted per original model's default
        # Custom CUDA function handle
        self.forward_cuda = conv_transpose_module.conv_transpose2d_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_cuda(
            x, self.weight,
            self.kernel_size[0], self.kernel_size[1],
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1]
        )