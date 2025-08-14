import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv1d_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv1d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits> output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_length,
    int output_length,
    int stride,
    int padding,
    int dilation) {

    CUDA_1D_KERNEL_LOOP(output_index, batch_size * out_channels * output_length) {
        int batch = output_index / (out_channels * output_length);
        int oc = (output_index / output_length) % out_channels;
        int ow = output_index % output_length;

        scalar_t val = 0;
        for (int kh = 0; kh < kernel_size; ++kh) {
            int iw = ow * stride + kh * dilation - padding;
            if (iw < 0 || iw >= input_length) continue;
            for (int ic = 0; ic < in_channels; ++ic) {
                val += weight[oc][ic][kh] * input[batch][ic][iw];
            }
        }
        output[batch][oc][ow] = val;
    }
}

torch::Tensor conv1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    int output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    const int threads = 1024;
    const int elements = batch_size * out_channels * output_length;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv1d_forward", ([&] {
        conv1d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,3,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            input_length,
            output_length,
            stride,
            padding,
            dilation);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv1d_cpp_source = (
    "torch::Tensor conv1d_forward(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"
)

conv1d_op = load_inline(
    name="conv1d_custom",
    cpp_sources=[conv1d_cpp_source],
    cuda_sources=[conv1d_source],
    functions=["conv1d_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv1d_op.conv1d_forward(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        return output