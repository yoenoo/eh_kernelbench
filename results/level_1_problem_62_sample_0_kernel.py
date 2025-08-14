import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv2d
conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Define the convolution kernel
__global__ void conv2d_kernel(const float* input, const float* weight, float* output,
                             int batch_size, int in_channels, int out_channels,
                             int input_height, int input_width, int kernel_height,
                             int kernel_width, int stride, int padding) {
    const int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    const int output_width = (input_width - kernel_width + 2 * padding * 2) / stride + 1;

    // Get output element indices
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int h_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int n = blockIdx.z / out_channels;
    int c_out = blockIdx.z % out_channels;

    if (w_idx >= output_width || h_idx >= output_height) {
        return;
    }

    // Compute input region boundaries
    int h_start = h_idx * stride - padding;
    int w_start = w_idx * stride - padding;
    int h_end = h_start + kernel_height;
    int w_end = w_idx * stride + kernel_width;

    // Accumulate the output value
    float acc = 0.0;
    for (int i = 0; i < kernel_height; ++i) {
        for (int j = 0; j < kernel_width; ++j) {
            int h = h_start + i;
            int w = w_start + j;
            // Check if the input coordinates are valid
            if (h < 0 || h >= input_height || w < 0 || w >= input_width) {
                continue;
            }
            for (int c_in = 0; c_in < in_channels; ++c_in) {
                acc += input[n * in_channels * input_height * input_width +
                            c_in * input_height * input_width +
                            h * input_width + w] *
                       weight[c_out * in_channels * kernel_height * kernel_width +
                              c_in * kernel_height * kernel_width +
                              i * kernel_width + j];
            }
        }
    }
    output[n * out_channels * output_height * output_width +
           c_out * output_height * output_width +
           h_idx * output_width + w_idx] = acc;
}

torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, 
                            int kernel_height, int kernel_width, int stride, int padding) {
    // Get the input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    // Compute output dimensions
    int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;
    int out_channels = weight.size(0);

    // Output tensor
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                               input.options());

    dim3 threads(16, 16);
    dim3 blocks((output_width + threads.x - 1) / threads.x,
               (output_height + threads.y - 1) / threads.y,
               batch_size * out_channels);

    conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width, kernel_height, kernel_width, stride, padding);

    cudaDeviceSynchronize();
    return output;
}
"""

conv2d_cuda_cpp_source = """
torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, 
                            int kernel_height, int kernel_width, int stride, int padding);
"""

conv2d_module = load_inline(
    name="conv2d_cuda",
    cpp_sources=conv2d_cuda_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_forward"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights similar to PyTorch's Conv2d
        kernel_height, kernel_width = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups,
                                               kernel_height, kernel_width))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

        # Bind the custom conv2d kernel
        self.forward_conv = conv2d_module.conv2d_forward

    def forward(self, x):
        output = self.forward_conv(x, self.weight.view(self.weight.size(0), -1), 
                                  self.kernel_size[0], self.kernel_size[1],
                                  self.stride, self.padding)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output