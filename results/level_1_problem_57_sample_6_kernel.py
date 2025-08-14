import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(const torch::PackedTensorAccessor<scalar_t,4> input,
                                       const torch::PackedTensorAccessor<scalar_t,4> weight,
                                       torch::PackedTensorAccessor<scalar_t,4> output,
                                       int kernel_size,
                                       int stride,
                                       int padding,
                                       int output_padding,
                                       int groups) {

    int batch_idx = blockIdx.x;
    int out_channel = blockIdx.y;
    int out_h = threadIdx.x;
    int out_w = threadIdx.y;

    scalar_t val = 0;
    for (int in_channel_group = 0; in_channel_group < groups; ++in_channel_group) {
        int in_channel_offset = in_channel_group * (input.size(1)/groups);
        int weight_channel = out_channel + in_channel_group * (weight.size(0)/groups);

        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int in_h = (out_h + 2 * padding - kh) / stride;
                int in_w = (out_w + 2 * padding - kw) / stride;

                if ((out_h + 2 * padding - kh) % stride == 0 && (out_w + 2 * padding - kw) % stride == 0 &&
                    in_h >= 0 && in_h < input.size(2) && 
                    in_w >= 0 && in_w < input.size(3)) {

                    for (int i_c = in_channel_offset; i_c < in_channel_offset + (input.size(1)/groups); ++i_c) {
                        val += input[batch_idx][i_c][in_h][in_w] * 
                               weight[weight_channel][i_c - in_channel_offset][kh][kw];
                    }
                }
            }
        }
    }
    output[batch_idx][out_channel][out_h][out_w] = val;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding, int output_padding, int groups) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_h = input.size(2);
    const auto in_w = input.size(3);

    const auto out_channels = weight.size(0);
    const auto out_h = (in_h - 1) * stride - 2 * padding + kernel_size + output_padding;
    const auto out_w = (in_w - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

    dim3 threads(32, 32); // thread block size
    dim3 blocks(batch_size, out_channels);

    const int max_threads = 1024;
    if (threads.x * threads.y > max_threads) {
        threads.x = sqrt(max_threads);
        threads.y = sqrt(max_threads);
    }

    conv_transpose2d_kernel<float><<<blocks, threads>>>(
        input.packed_accessor<float,4>(),
        weight.packed_accessor<float,4>(),
        output.packed_accessor<float,4>(),
        kernel_size,
        stride,
        padding,
        output_padding,
        groups
    );

    return output;
}
"""

conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding, int output_padding, int groups);"
)

# Compile the CUDA kernels
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize parameters similar to ConvTranspose2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv_transpose2d.conv_transpose2d_cuda(x, self.weight, self.kernel_size, self.stride, self.padding, self.output_padding, self.groups)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output