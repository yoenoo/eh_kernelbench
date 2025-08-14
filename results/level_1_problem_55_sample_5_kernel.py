import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized 2D convolution
conv2d_kernel = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv2d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,4> input,
    torch::PackedTensorAccessor<scalar_t,4> output,
    torch::PackedTensorAccessor<scalar_t,4> weight,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    int out_h = (input.size[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (input.size[3] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    int N = input.size[0];
    int in_c = input.size[1];
    int out_c = weight.size[0];

    int batch_idx = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = blockIdx.z;

    if (batch_idx < N && out_y < out_h && out_x < out_w) {
        scalar_t sum = 0;
        for (int k = 0; k < in_c; ++k) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int in_y = out_y * stride + ky * dilation - padding;
                    int in_x = out_x * stride + kx * dilation - padding;
                    if (in_y >= 0 && in_y < input.size[2] && in_x >= 0 && in_x < input.size[3]) {
                        sum += input[batch_idx][k][in_y][in_x] * weight[out_c * batch_idx + k][k][ky][kx];
                    }
                }
            }
        }
        output[batch_idx][out_y][out_x] = sum;
    }
}

std::tuple<torch::Tensor> conv2d_forward(
        torch::Tensor input,
        torch::Tensor weight,
        int kernel_size,
        int stride,
        int padding,
        int dilation) {

    const auto output = torch::zeros({input.size(0), weight.size(0),
        (input.size(2) + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1,
        (input.size(3) + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1},
        input.options());

    dim3 blocks(input.size(0), output.size(2), output.size(3));
    dim3 threads(1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_forward", ([&] {
        conv2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    return output;
}
"""

conv2d_cpp_source = """
std::tuple<torch::Tensor> conv2d_forward(
        torch::Tensor input,
        torch::Tensor weight,
        int kernel_size,
        int stride,
        int padding,
        int dilation);
"""

# Compile the inline CUDA code for convolution
conv2d = load_inline(
    name="conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_kernel,
    functions=["conv2d_forward"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-gencode=arch=compute_80,code=sm_80"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights using He initialization
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size) * (2.0 / (in_channels * kernel_size * kernel_size))**0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.conv2d_cuda = conv2d

    def forward(self, x):
        output = self.conv2d_cuda.conv2d_forward(x, self.weight, self.kernel_size, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output