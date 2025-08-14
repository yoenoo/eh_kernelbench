import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for depthwise convolution
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__forceinline__ __device__ float calc_depthwise_conv(float* input, float* kernel, int kernel_size, int channels, int in_height, int in_width, int out_height, int out_width, int batch_idx, int channel_idx, int out_y, int out_x, int padding, int stride) {
    float result = 0.0;
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            int in_y = out_y * stride + ky - padding;
            int in_x = out_x * stride + kx - padding;
            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                result += input[batch_idx * channels * in_height * in_width + channel_idx * in_height * in_width + (in_y * in_width + in_x)] * 
                         kernel[channel_idx * kernel_size * kernel_size + (ky * kernel_size + kx)];
            }
        }
    }
    return result;
}

__global__ void depthwise_conv2d_kernel(float* input, float* kernel, float* output, int batch_size, int in_channels, int in_height, int in_width, int kernel_size, int out_height, int out_width, int padding, int stride) {
    int batch = blockIdx.x;
    int channel = blockIdx.y;
    int out_y = threadIdx.y;
    int out_x = threadIdx.x + blockIdx.z * blockDim.x;

    if (out_x >= out_width) return;

    float res = calc_depthwise_conv(input, kernel, kernel_size, in_channels, in_height, in_width, out_height, out_width, batch, channel, out_y, out_x, padding, stride);
    output[batch * in_channels * out_height * out_width + channel * out_height * out_width + out_y * out_width + out_x] = res;
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor kernel, int kernel_size, int stride, int padding) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, in_channels, out_height, out_width}, input.options());

    int block_dim_x = 32;
    int grid_dim_x = (out_width + block_dim_x - 1) / block_dim_x;

    dim3 threads(block_dim_x, 16); // threads per block
    dim3 grid(batch_size, in_channels, grid_dim_x); // blocks per grid

    depthwise_conv2d_kernel<<<grid, threads>>>(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        kernel_size,
        out_height,
        out_width,
        padding,
        stride
    );

    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor kernel, int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code for depthwise convolution
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Initialize weight and kernel similar to Conv2d
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # He initialization

        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.weight.view(self.in_channels, self.kernel_size * self.kernel_size)
        output = depthwise_conv2d.depthwise_conv2d_cuda(x, kernel, self.kernel_size, self.stride, self.padding)
        
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        return output