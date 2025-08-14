import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d
conv_transpose_2d_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < n;          \
      i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose2d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int in_channels, int out_channels, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int dilation_h, int dilation_w,
    int groups)
{
    // Your custom implementation here
    // This part requires careful kernel implementation of the transposed convolution
    // ...

    const int batch_size = input.size(0);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    CUDA_1D_KERNEL_LOOP(index, batch_size * out_h * out_w) {
        int w_out = index % out_w;
        int h_out = (index / out_w) % out_h;
        int n = index / (out_h * out_w);

        for (int c_out = 0; c_out < out_channels; ++c_out) {
            scalar_t val = 0;
            for (int k_h = 0; k_h < kernel_h; ++k_h) {
                for (int k_w = 0; k_w < kernel_w; ++k_w) {
                    // Compute the input position
                    int h_in = (h_out - k_h * dilation_h - output_padding_h) / stride_h + padding_h;
                    int w_in = (w_out - k_w * dilation_w - output_padding_w) / stride_w + padding_w;

                    // Check if the input position is valid
                    if (h_in >= 0 && h_in < input.size(2) && w_in >= 0 && w_in < input.size(3)) {
                        for (int c_in_group = 0; c_in_group < in_channels/groups; ++c_in_group) {
                            int c_in = c_in_group + (c_out / (out_channels/groups)) * in_channels/groups;
                            val += input[n][c_in][h_in][w_in] * weight[c_out][c_in_group][k_h][k_w];
                        }
                    }
                }
            }
            output[n][c_out][h_out][w_out] = val;
        }
    }
}

torch::Tensor conv_transpose2d_forward(torch::Tensor input, torch::Tensor weight,
    int stride_h, int stride_w, int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int dilation_h, int dilation_w,
    int groups)
{
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_h = input.size(2);
    const auto in_w = input.size(3);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);
    const auto out_channels = weight.size(0);

    // Calculate output dimensions
    auto out_h = (in_h - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1;
    auto out_w = (in_w - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, input.options());

    int blocks = (batch_size * out_h * out_w + 1024 - 1) / 1024;
    dim3 grid(blocks);
    dim3 block(1024);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_forward", ([&] {
        conv_transpose2d_forward_kernel<scalar_t><<<grid, block>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            in_channels, out_channels, kernel_h, kernel_w,
            stride_h, stride_w, padding_h, padding_w,
            output_padding_h, output_padding_w,
            dilation_h, dilation_w,
            groups
        );
    }));

    return output;
}
"""

conv_transpose_2d_cpp_source = (
    "torch::Tensor conv_transpose2d_forward(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w, int output_padding_h, int output_padding_w, int dilation_h, int dilation_w, int groups);"
)

# Compile the custom CUDA kernel
conv_transpose_2d = load_inline(
    name="conv_transpose_2d",
    cpp_sources=conv_transpose_2d_cpp_source,
    cuda_sources=conv_transpose_2d_source,
    functions=["conv_transpose2d_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

        # Initialize weight and bias similar to PyTorch's ConvTranspose2d
        kh, kw = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kh, kw))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        output_padding_h, output_padding_w = self.output_padding
        dilation_h, dilation_w = self.dilation

        # Call the custom CUDA kernel
        output = conv_transpose_2d.conv_transpose2d_forward(
            x,
            self.weight,
            stride_h, stride_w,
            padding_h, padding_w,
            output_padding_h, output_padding_w,
            dilation_h, dilation_w,
            self.groups
        )

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output