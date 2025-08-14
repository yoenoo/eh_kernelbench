import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv1d_relu_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv1d_relu_kernel(const scalar_t* __restrict__ input,
                                  const scalar_t* __restrict__ weight,
                                  scalar_t* __restrict__ output,
                                  int batch, int in_channels, int out_channels,
                                  int kernel_size, int length, int length_out,
                                  int stride, int padding, int dilation) {
    CUDA_1D_KERNEL_LOOP(output_idx, batch * out_channels * length_out) {
        int batch_idx = output_idx / (out_channels * length_out);
        int out_channel = (output_idx / length_out) % out_channels;
        int out_pos = output_idx % length_out;

        scalar_t sum = 0;
        for (int k = 0; k < kernel_size; ++k) {
            int in_pos = out_pos * stride + k * dilation - padding;
            if (in_pos < 0 || in_pos >= length) continue;
            for (int c = 0; c < in_channels; ++c) {
                sum += input[batch_idx * in_channels * length + c * length + in_pos] *
                       weight[out_channel * in_channels * kernel_size + c * kernel_size + k];
            }
        }
        output[output_idx] = fmaxf(sum, static_cast<scalar_t>(0)); // Apply ReLU
    }
}

at::Tensor conv1d_relu_cuda(at::Tensor input, at::Tensor weight, int stride, int padding, int dilation) {
    const int batch = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int length = input.size(2);
    const int length_out = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = at::empty({batch, out_channels, length_out}, input.options());

    const int threads = 256;
    const int elements = batch * out_channels * length_out;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv1d_relu_cuda", ([&] {
        conv1d_relu_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(), weight.data<scalar_t>(),
            output.data<scalar_t>(), batch, in_channels, out_channels,
            kernel_size, length, length_out, stride, padding, dilation);
    }));

    return output;
}
"""

conv1d_relu_header = """
at::Tensor conv1d_relu_cuda(at::Tensor input, at::Tensor weight, int stride, int padding, int dilation);
"""

conv1d_relu = load_inline(
    name='conv1d_relu',
    cpp_sources=conv1d_relu_header,
    cuda_sources=conv1d_relu_source,
    functions=['conv1d_relu_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # For simplicity, assuming groups=1 and no bias for the optimized kernel
        assert groups == 1, "Only groups=1 supported in this optimized version"
        assert not bias, "Bias not implemented in this optimized version"
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize weights similar to PyTorch's default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return conv1d_relu.conv1d_relu_cuda(x, self.weight, self.stride, self.padding, self.dilation)