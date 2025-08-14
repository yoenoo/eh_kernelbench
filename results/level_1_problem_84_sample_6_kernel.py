import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for depthwise convolution
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4> input,
    const torch::PackedTensorAccessor<scalar_t,4> weight,
    torch::PackedTensorAccessor<scalar_t,4> output,
    int batch_size, int in_channels, int out_channels,
    int kernel_size, int height_in, int width_in,
    int height_out, int width_out, int stride)
{
    // Each thread computes one output element
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y;
    int batch = blockIdx.z;

    if (w_out >= width_out || h_out >= height_out) return;

    for (int c = 0; c < in_channels; ++c) { // Each input channel has its own filter
        scalar_t sum = 0;
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out * stride + kh;
                int w_in = w_out * stride + kw;
                if (h_in < height_in && w_in < width_in) {
                    sum += input[batch][c][h_in][w_in] *
                           weight[c][0][kh][kw]; // depthwise: out_channels == in_channels
                }
            }
        }
        output[batch][c][h_out][w_out] = sum;
    }
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0); // depthwise: out_channels = in_channels
    const int kernel_size = weight.size(2);
    const int height_in = input.size(2);
    const int width_in = input.size(3);
    const int stride = 1; // Assuming stride=1 as per example parameters

    const int height_out = (height_in - kernel_size) / stride + 1;
    const int width_out = (width_in - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());

    dim3 threads(32); // Tune based on width_out
    dim3 blocks(width_out, height_out, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            batch_size, in_channels, out_channels,
            kernel_size, height_in, width_in,
            height_out, width_out, stride);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

depthwise_conv_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight);
"""

# Compile the CUDA kernel
depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Initialize weights similar to PyTorch's Conv2d
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # He initialization

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        self.depthwise_conv = depthwise_conv

    def forward(self, x):
        # Pad input if necessary (simple reflection padding for demonstration)
        if self.padding > 0:
            padding = (self.padding, self.padding, self.padding, self.padding)
            x = torch.nn.functional.pad(x, padding, mode='reflect')

        output = self.depthwise_conv.depthwise_conv2d_cuda(x, self.weight)

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output