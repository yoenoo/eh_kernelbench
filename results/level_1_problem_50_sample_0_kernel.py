import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Convolution with kernel_size 11, stride 4, padding 2
conv2d_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_1D_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv2d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
                                     torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
                                     torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
                                     int kernel_size, int stride, int padding) {
    const int B = output.size(0);
    const int M = output.size(1);
    const int OH = output.size(2);
    const int OW = output.size(3);

    CUDA_1D_KERNEL_LOOP(index, B * M * OH * OW) {
        int ow = index % OW;
        int oh = (index / OW) % OH;
        int m = (index / (OW * OH)) % M;
        int b = index / (M * OW * OH);

        scalar_t val = 0;
        for (int c = 0; c < input.size(1); ++c) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int h_in = oh * stride + kh - padding;
                    int w_in = ow * stride + kw - padding;
                    // skip if out of input boundaries
                    if (h_in >= 0 && h_in < input.size(2) && w_in >=0 && w_in < input.size(3)) {
                        val += input[b][c][h_in][w_in] * weight[m][c][kh][kw];
                    }
                }
            }
        }
        output[b][m][oh][ow] = val;
    }
}

torch::Tensor custom_conv2d_forward(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding) {
    auto output = torch::zeros({input.size(0), weight.size(0), 
                              (input.size(2) + 2 * padding - kernel_size) / stride + 1,
                              (input.size(3) + 2 * padding - kernel_size) / stride + 1},
                              input.options());

    int threads = 256;
    int num_elements = output.numel();
    int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_forward", ([&]{
        conv2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            kernel_size, stride, padding);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv2d_cpp_source = "torch::Tensor custom_conv2d_forward(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding);"

# Compile custom convolution kernel
conv2d_module = load_inline(
    name="custom_conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_kernel_source,
    functions=["custom_conv2d_forward"],
    verbose=True,
    extra_cflags=["-DVERSION_GE_1_5"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        # Define convolution parameters
        self.in_channels = 3
        self.out_channels = 96
        self.kernel_size = 11
        self.stride = 4
        self.padding = 2

        # Initialize weights similar to PyTorch's Conv2d initialization
        weight = torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        return conv2d_module.custom_conv2d_forward(x, self.weight, self.kernel_size, self.stride, self.padding)