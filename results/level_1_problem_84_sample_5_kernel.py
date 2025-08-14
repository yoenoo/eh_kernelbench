import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom Depthwise Convolution CUDA Kernel
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
                                       const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
                                       torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
                                       int batch_size, int in_channels, int out_channels,
                                       int kernel_size, int height_in, int width_in, int stride,
                                       int padding, int height_out, int width_out) {

    int N = blockIdx.x;
    int C = blockIdx.y;
    int y_out = threadIdx.y;
    int x_out = threadIdx.x;

    // Compute output dimensions
    int y_in = -padding + y_out * stride;
    int x_in = -padding + x_out * stride;

    scalar_t sum = 0;
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            int y = y_in + ky;
            int x = x_in + kx;
            // Check if the current kernel position is within the image bounds
            if (y >= 0 && y < height_in && x >= 0 && x < width_in) {
                sum += input[N][C][y][x] * weight[C][0][ky][kx];
            }
        }
    }
    output[N][C][y_out][x_out] = sum;
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int stride, int padding) {
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height_in = input.size(2);
    int width_in = input.size(3);
    int kernel_size = weight.size(2);
    int out_channels = weight.size(0);
    int height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    int width_out = (width_in + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());

    dim3 threads(kernel_size, kernel_size);
    dim3 blocks(batch_size, out_channels);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, in_channels, out_channels, kernel_size,
            height_in, width_in, stride, padding,
            height_out, width_out);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                                  int stride, int padding);
"""

depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Initialize weights for depthwise convolution
        self.weight = nn.Parameter(torch.randn(out_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        output = depthwise_conv2d.depthwise_conv2d_cuda(x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output