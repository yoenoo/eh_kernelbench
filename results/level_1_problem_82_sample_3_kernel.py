import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREADS 256

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ weight,
                                       scalar_t* __restrict__ output,
                                       int batch_size,
                                       int in_channels,
                                       int input_height,
                                       int input_width,
                                       int kernel_size,
                                       int output_height,
                                       int output_width) {
    int in_channel = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = threadIdx.x;

    if (out_x >= output_width) return;

    const int input_offset = in_channel * input_height * input_width;
    const int output_offset = in_channel * output_height * output_width;

    scalar_t acc = 0;
    for (int ky = 0; ky < kernel_size; ++ky) {
        int in_y = out_y + ky;
        if (in_y >= input_height) continue;
        for (int kx = 0; kx < kernel_size; ++kx) {
            int in_x = out_x + kx;
            if (in_x >= input_width) continue;
            const scalar_t w = weight[ky * kernel_size + kx];
            const scalar_t i = input[input_offset + in_y * input_width + in_x];
            acc += w * i;
        }
    }
    output[output_offset + out_y * output_width + out_x] = acc;
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int kernel_size,
                                   int output_height,
                                   int output_width) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);

    auto output = torch::empty({batch_size, in_channels, output_height, output_width}, input.options());

    dim3 threads_per_block(THREADS);
    dim3 blocks_per_grid(in_channels, output_height);

    const int kernel_radius = kernel_size / 2;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks_per_block, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            input_height,
            input_width,
            kernel_size,
            output_height,
            output_width);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

depthwise_conv_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int kernel_size,
                                   int output_height,
                                   int output_width);
"""

depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty((kernel_size * kernel_size)))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        self.reset_parameters()

        # Initialize weights similar to PyTorch's default (He initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.depthwise_conv = depthwise_conv

    def forward(self, x):
        # Compute output dimensions
        input_height, input_width = x.size(2), x.size(3)
        output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Pad input if necessary
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        # Apply custom CUDA kernel
        output = self.depthwise_conv.depthwise_conv2d_cuda(
            x,
            self.weight.view(-1),
            self.kernel_size,
            output_height,
            output_width
        )

        # Handle bias (if any)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        return output