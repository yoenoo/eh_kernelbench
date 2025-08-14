import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed convolution
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

#define BLOCK_SIZE 512

__global__ void conv_transpose2d_kernel(
    const float* input, const float* weight, float* output,
    int batch_size, int in_channels, int out_channels, int kernel_size,
    int input_h, int input_w, int output_h, int output_w,
    int stride, int padding, int output_padding, int groups) {

    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * out_channels * output_h * output_w) {
        return;
    }

    int w_out = output_idx % output_w;
    int h_out = (output_idx / output_w) % output_h;
    int c_out = (output_idx / (output_w * output_h)) % out_channels;
    int n = output_idx / (out_channels * output_h * output_w);

    float acc = 0.0;
    for (int k_h = 0; k_h < kernel_size; ++k_h) {
        for (int k_w = 0; k_w < kernel_size; ++k_w) {
            int in_h = (h_out + padding - k_h) / stride;
            int in_w = (w_out + padding - k_w) / stride;

            if ((h_out + padding - k_h) % stride == 0 && (w_out + padding - k_w) % stride == 0 &&
                in_h >= 0 && in_h < input_h &&
                in_w >= 0 && in_w < input_w) {

                for (int g = 0; g < groups; ++g) {
                    int c_in = (c_out / (out_channels / groups)) * (in_channels / groups) + g;
                    acc += input[n * in_channels * input_h * input_w + c_in * input_h * input_w + in_h * input_w + in_w] *
                           weight[c_out * kernel_size * kernel_size * in_channels/(groups) + (k_h * kernel_size + k_w) * in_channels/groups + (c_in % (in_channels/groups))];
                }
            }
        }
    }

    output[output_idx] = acc;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding,
    int groups) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = sqrt(weight.size(1) / in_channels);

    // Calculate output dimensions
    int output_h = (input_h - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_w = (input_w - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());

    int total_outputs = batch_size * out_channels * output_h * output_w;
    int block_size = BLOCK_SIZE;
    int grid_size = (total_outputs + block_size - 1) / block_size;

    conv_transpose2d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, kernel_size,
        input_h, input_w, output_h, output_w,
        stride, padding, output_padding, groups);

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int output_padding, int groups);"
)

conv_transpose2d_module = load_inline(
    name="conv_transpose2d_cuda",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-g"],
    extra_ldflags=["-g"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weights like nn.ConvTranspose2d
        weight_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv_transpose2d_module.conv_transpose2d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output