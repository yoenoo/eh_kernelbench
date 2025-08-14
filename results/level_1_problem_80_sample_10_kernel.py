import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv2d
conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv2d_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output,
    int kernel_h, int kernel_w,
    int stride, int padding_h, int padding_w,
    int dilation_h, int dilation_w) {

    int batch = blockIdx.x;
    int out_channel = blockIdx.y;
    int out_h = threadIdx.y;
    int out_w = threadIdx.x;

    scalar_t sum = 0;
    for (int k_h = 0; k_h < kernel_h; ++k_h) {
        for (int k_w = 0; k_w < kernel_w; ++k_w) {
            int in_h = out_h * stride + k_h * dilation_h - padding_h;
            int in_w = out_w * stride + k_w * dilation_w - padding_w;
            if (in_h >= 0 && in_h < input.size(2) && in_w >= 0 && in_w < input.size(3)) {
                for (int c = 0; c < input.size(1); ++c) {
                    sum += input[batch][c][in_h][in_w] * weight[out_channel][c][k_h][k_w];
                }
            }
        }
    }
    output[batch][out_channel][out_h][out_w] = sum;
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, std::pair<int, int> padding, std::pair<int, int> dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_h = (in_h + 2*padding.first - dilation.first*(kernel_h - 1) - 1)/stride + 1;
    const int out_w = (in_w + 2*padding.second - dilation.second*(kernel_w - 1) - 1)/stride + 1;

    torch::Tensor output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

    dim3 threads(out_w, out_h);
    dim3 blocks(batch_size, out_channels);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_cuda", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            kernel_h, kernel_w,
            stride, padding.first, padding.second,
            dilation.first, dilation.second);
    }));

    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, std::pair<int, int> padding, std::pair<int, int> dilation);"
)

# Compile the inline CUDA code for Conv2d
conv2d = load_inline(
    name="conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=False,
    extra_cflags=["-DWITH_CUDA", "-xcuda"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=(0,0), dilation=(1,1), bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        # Initialize weights using a normal distribution for testing
        nn.init.normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.normal_(self.bias)
        else:
            self.bias = None
        self.conv2d = conv2d  # the custom CUDA function

    def forward(self, x):
        output = self.conv2d.conv2d_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output