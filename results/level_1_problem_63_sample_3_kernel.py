import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_relu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for fused Conv2D + ReLU
__global__ void conv_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int kernel_size,
    int out_height, int out_width, int stride, int padding) {

    const int output_depth = out_channels;
    const int data_width = in_width;
    const int data_height = in_height;
    const int filt_width = kernel_size;
    const int pad = padding;

    int n = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = blockIdx.z;
    int c_out = threadIdx.x;

    if (c_out >= output_depth) return;

    int in_x_origin = out_x * stride - pad;
    int in_y_origin = out_y * stride - pad;

    float acc = (bias) ? bias[c_out] : 0.0f;

    for (int ff = 0; ff < filt_width; ++ff) {
        for (int cc = 0; cc < in_channels; ++cc) {
            for (int jj = 0; jj < filt_width; ++jj) {
                int in_x = in_x_origin + jj;
                int in_y = in_y_origin + ff;
                if (in_x >= 0 && in_x < data_width && in_y >= 0 && in_y < data_height) {
                    int w_idx = (c_out * in_channels + cc) * (filt_width*filt_width) + ff*filt_width + jj;
                    int i_idx = n * in_channels * data_height * data_width + cc * data_height * data_width + in_y * data_width + in_x;
                    acc += weight[w_idx] * input[i_idx];
                }
            }
        }
    }

    output[((n * output_depth + c_out) * out_height + out_y) * out_width + out_x] = fmaxf(acc, 0.0f);
}

torch::Tensor conv_relu_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_height = input.size(2);
    const auto in_width = input.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);
    const auto out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const auto out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());

    dim3 threads(out_channels); // Each thread handles one output channel
    dim3 blocks(batch_size, out_height, out_width);

    conv_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        (bias.defined() ? bias.data_ptr<float>() : nullptr),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_height, in_width, kernel_size,
        out_height, out_width, stride, padding);

    cudaDeviceSynchronize();
    return output;
}
"""

# Compile fused kernel
conv_relu_cpp_source = "torch::Tensor conv_relu_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding);"
conv_relu = load_inline(name="conv_relu", cpp_sources=conv_relu_cpp_source, cuda_sources=conv_relu_source, functions=["conv_relu_forward"], verbose=True)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels//groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x):
        return conv_relu.conv_relu_forward(x, self.weight, self.bias if hasattr(self, 'bias') else torch.Tensor(), self.stride, self.padding)