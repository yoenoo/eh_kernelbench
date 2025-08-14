import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 1D convolution
conv_transpose_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose_1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_size,
    const int length_out,
    const int length_in,
    const int stride,
    const int padding,
    const int dilation) {

    CUDA_1D_KERNEL_LOOP(output_index, batch_size * out_channels * length_out) {
        int batch = output_index / (out_channels * length_out);
        int out_channel = (output_index / length_out) % out_channels;
        int out_pos = output_index % length_out;

        scalar_t value = 0;
        for (int kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
            int in_channel = out_channel;
            int weight_offset = in_channel * kernel_size + kernel_idx;
            int input_rel_pos = out_pos + (kernel_idx * dilation) - padding;
            if (input_rel_pos < 0 || input_rel_pos >= length_in) {
                continue;
            }
            int input_abs_pos = batch * in_channels * length_in + in_channel * length_in + input_rel_pos;
            int weight_pos = out_channel * in_channels * kernel_size + (kernel_idx * in_channels + in_channel);
            value += input[input_abs_pos] * weight[weight_pos];
        }
        output[output_index] = value;
    }
}

torch::Tensor conv_transpose_1d_cuda(torch::Tensor input, torch::Tensor weight,
                                    int stride, int padding, int dilation) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    auto length_in = input.size(2);
    auto length_out = (length_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, length_out}, input.options());

    dim3 threads_per_block(256);
    dim3 num_blocks((output.numel() + threads_per_block.x - 1) / threads_per_block.x);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose_1d_cuda", ([&] {
        conv_transpose_1d_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, out_channels,
            kernel_size, length_out, length_in,
            stride, padding, dilation);
    }));

    return output;
}
"""

conv_transpose_1d_cpp_source = (
    "torch::Tensor conv_transpose_1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"
)

# Compile the custom CUDA kernel
conv_transpose_1d = load_inline(
    name="conv_transpose_1d",
    cpp_sources=[conv_transpose_1d_cpp_source],
    cuda_sources=[conv_transpose_1d_source],
    functions=["conv_transpose_1d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        # Initialize weights similar to PyTorch's ConvTranspose1d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None
        # Register the custom CUDA kernel
        self.custom_conv_transpose = conv_transpose_1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform convolution using custom kernel
        output = self.custom_conv_transpose.conv_transpose_1d_cuda(
            x, self.weight, self.stride, self.padding, self.dilation
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
        return output