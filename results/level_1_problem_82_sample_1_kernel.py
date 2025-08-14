import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(const torch::PackedTensorAccessor<scalar_t,4> input,
                                       const torch::PackedTensorAccessor<scalar_t,4> weight,
                                       torch::PackedTensorAccessor<scalar_t,4> output,
                                       int batch_size, int channels,
                                       int in_h, int in_w,
                                       int kernel_size,
                                       int stride,
                                       int padding) {

    const int H = output.size(2);
    const int W = output.size(3);

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    const int channel = blockIdx.z;
    const int batch = blockIdx.w;

    if (channel >= channels || batch >= batch_size) return;

    int out_x = col;
    int out_y = row;

    scalar_t sum = 0;
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            int in_row = out_y * stride + ky - padding;
            int in_col = out_x * stride + kx - padding;
            
            // Skip out-of-bound indices
            if (in_row >= 0 && in_row < in_h && in_col >=0 && in_col < in_w) {
                sum += input[batch][channel][in_row][in_col] * 
                       weight[channel][0][ky][kx];
            }
        }
    }

    output[batch][channel][row][col] = sum;
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int stride, int padding) {

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto in_h = input.size(2);
    const auto in_w = input.size(3);
    const auto kernel_size = weight.size(2);

    // Compute output dimensions
    auto out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    auto out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, channels, out_h, out_w}, input.options());

    const int threads = 256;
    dim3 blocks(out_w, out_h, channels, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            batch_size, channels, in_h, in_w,
            kernel_size, stride, padding);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

depthwise_conv_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding);
"""

depthwise_conv = load_inline(
    name='depthwise_conv',
    cpp_sources=[depthwise_conv_cpp_source],
    cuda_sources=[depthwise_conv_source],
    functions=['depthwise_conv2d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights similar to PyTorch's Conv2d
        self.weight = nn.Parameter(
            torch.empty(in_channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        output = depthwise_conv.depthwise_conv2d_cuda(
            x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output