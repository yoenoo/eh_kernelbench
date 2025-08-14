import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized 3D convolution
conv3d_custom_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv3d_kernel(const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> input,
                        const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> weight,
                        torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> output,
                        int batch_size, int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation, int depth)
{
    const int output_depth = depth;
    const int output_height = (in_channels + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = output_height; // Assuming square dimensions
    const int h_idx = blockIdx.z;
    const int w_idx = blockIdx.y;
    const int b_idx = blockIdx.x;

    const int out_channel = threadIdx.z;
    const int in_channel_group = threadIdx.y;
    const int group_idx = threadIdx.x;

    const int in_channel = in_channel_group * blockDim.x + group_idx;

    scalar_t val = 0;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_in = h_idx * stride - padding + dilation * kh;
            int w_in = w_idx * stride - padding + dilation * kw;
            if (h_in >= 0 && h_in < in_channels && w_in >= 0 && w_in < in_channels) {
                val += input[b_idx][in_channel][h_in][w_in][0] * weight[out_channel][in_channel][kh][kw][0];
            }
        }
    }
    output[b_idx][out_channel][h_idx][w_idx][0] = val;
}

int elements(int volume) {
    return volume * volume * volume;
}

torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation)
{
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);
    const auto depth = input.size(4);

    const int output_height = (in_channels + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = output_height;
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width, depth}, input.options());

    dim3 threads(32, 8, 8); // Threads per block
    dim3 blocks(output_width, output_height, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_forward_cuda", ([&] {
        using scalar_t = at::acc_type<scalar_t, true>;
        conv3d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
            weight.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
            batch_size, in_channels, out_channels, kernel_size, stride, padding, dilation, depth);
    }));

    return output;
}
"""

conv3d_custom_cpp_source = (
    "torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"
)

# Compile the custom kernel
conv3d_custom = load_inline(
    name="conv3d_custom",
    cpp_sources=conv3d_custom_cpp_source,
    cuda_sources=conv3d_custom_source,
    functions=["conv3d_forward_cuda"],
    verbose=True,
    extra_cflags=["-Dsupply='__CUDA_NO_HALF_OPERATORS__'"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, 1))
        self.bias_param = nn.Parameter(torch.empty(out_channels)) if bias else None

        # Initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x):
        return conv3d_custom.conv3d_forward_cuda(x, self.weight, self.stride, self.padding, self.dilation)