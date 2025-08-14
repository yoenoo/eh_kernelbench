import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose1d
conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;      \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose1d_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits> output,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int input_length,
    int output_length,
    int output_height) {

  CUDA_1D_KERNEL_LOOP(output_index, output_height * output_length * out_channels) {
    int output_channel = output_index / (output_height * output_length);
    int pos_out = (output_index / out_channels) % output_length;
    int n = output_index % output_height;

    scalar_t val = 0;
    for (int kw = 0; kw < kernel_size; ++kw) {
        const int d = kw * dilation;
        const int in_pos = pos_out - padding - d;
        if (in_pos < 0 || in_pos % stride != 0)
            continue;
        int input_pos = in_pos / stride;
        if (input_pos >= input_length || input_pos < 0)
            continue;
        for (int ich = 0; ich < in_channels; ++ich) {
            val += input[n][ich][input_pos] * weight[output_channel][ich][kw];
        }
    }
    output[n][output_channel][pos_out] = val;
  }
}

torch::Tensor conv_transpose1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation, int kernel_size) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_length = input.size(2);

    const int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + 2 * padding;
    const auto output_size = torch::IntArrayRef({batch_size, weight.size(0), output_length});
    auto output = torch::zeros(output_size, torch::device(input.device()).dtype(input.dtype()));

    const int threads = 256;
    const int elements = batch_size * output_length * weight.size(0);
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose1d_cuda", ([&] {
        conv_transpose1d_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,3,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits>(),
            in_channels,
            weight.size(0),
            kernel_size,
            stride,
            padding,
            dilation,
            input_length,
            output_length,
            batch_size);
    }));

    return output;
}
"""

conv_transpose1d_cpp_source = (
    "torch::Tensor conv_transpose1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation, int kernel_size);"
)

conv_transpose1d_module = load_inline(
    name="conv_transpose1d_cuda",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_cuda"],
    verbose=True,
    extra_cflags=["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__"],
    extra_cuda_cflags=["-lineinfo", "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        # Custom weights to mimic nn.ConvTranspose1d initialization
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        output = conv_transpose1d_module.conv_transpose1d_cuda(
            x, 
            self.weight, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.kernel_size
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
        return output

# Note: The original get_inputs and get_init_inputs functions remain unchanged for compatibility with testing framework
def get_inputs():
    x = torch.rand(batch_size, in_channels, length).cuda()
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]