import torch
import torch.nn as nn

from torch.utils.cpp_extension import load_inline

# Custom CUDA implementation of Conv1dTranspose
conv_transpose_1d_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose_1d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> output,
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t input_length,
    int64_t output_length,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    CUDA_1D_KERNEL_LOOP(output_index, batch_size * out_channels * output_length) {
        int batch = output_index / (out_channels * output_length);
        int residual = output_index % (out_channels * output_length);
        int oc = residual / output_length;
        int ol = residual % output_length;

        scalar_t val = 0;
        for (int ic_group = 0; ic_group < in_channels; ic_group++) {
            for (int kl = 0; kl < kernel_size; ++kl) {
                int il = ol * stride - padding - kl - output_padding;
                if (il < 0 || il >= input_length) continue;

                int kernel_idx = ic_group * kernel_size + kl;
                val += input[batch][ic_group][il] * weight[oc][ic_group][kl];
            }
        }
        output[batch][oc][ol] = val;
    }
}

torch::Tensor conv_transpose_1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_length = input.size(2);
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);
    const auto output_length = (input_length - 1) * stride + kernel_size - 2 * padding + output_padding;

    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    int blocks = (batch_size * out_channels * output_length + 1024 - 1) / 1024;
    conv_transpose_1d_forward_kernel<float>
        <<<blocks, 1024>>>(input.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                          weight.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                          output.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                          batch_size,
                          in_channels,
                          out_channels,
                          kernel_size,
                          input_length,
                          output_length,
                          stride,
                          padding,
                          output_padding);

    return output;
}
"""

conv_transpose_1d_cpp_source = "torch::Tensor conv_transpose_1d_forward(torch::Tensor input, torch::Tensor weight, int64_t stride, int64_t padding, int64_t output_padding);"

conv_transpose_1d_module = load_inline(
    name="conv_transpose_1d",
    cpp_sources=conv_transpose_1d_cpp_source,
    cuda_sources=conv_transpose_1d_source,
    functions="conv_transpose_1d_forward",
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weight like PyTorch's ConvTranspose1d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return conv_transpose_1d_module.conv_transpose_1d_forward(
            x, self.weight, self.stride, self.padding, self.output_padding
        )