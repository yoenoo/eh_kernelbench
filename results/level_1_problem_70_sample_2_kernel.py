import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Custom CUDA kernel for optimized 3D transpose convolution
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv_transpose_3d_kernel(
    const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> output,
    int out_depth, int out_height, int out_width,
    int in_depth, int in_height, int in_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    const int batch_idx = blockIdx.x;
    const int out_channel = blockIdx.y;
    const int out_d = threadIdx.z;
    const int out_h = threadIdx.y;
    const int out_w = threadIdx.x;

    const int in_group = out_channel / (weight.size(0) / groups);
    const int weight_channel = out_channel % (weight.size(0) / groups);

    scalar_t val = 0;
    for (int kd = 0; kd < kernel_depth; ++kd) {
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                const int in_d = (out_d - kd * dilation_d - padding_d) / stride_d;
                const int in_h = (out_h - kh * dilation_h - padding_h) / stride_h;
                const int in_w = (out_w - kw * dilation_w - padding_w) / stride_w;

                if (in_d >= 0 && in_d < in_depth &&
                    in_h >= 0 && in_h < in_height &&
                    in_w >= 0 && in_w < in_width) {
                    val += input[batch_idx][in_group * (weight.size(1)/groups) + weight_channel][in_d][in_h][in_w] *
                           weight[out_channel][weight_channel][kd][kh][kw];
                }
            }
        }
    }
    output[batch_idx][out_channel][out_d][out_h][out_w] = val;
}

torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight, 
                                    int stride_d, int stride_h, int stride_w,
                                    int padding_d, int padding_h, int padding_w,
                                    int output_padding_d, int output_padding_h, int output_padding_w,
                                    int dilation_d, int dilation_h, int dilation_w,
                                    int groups) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_depth = input.size(2);
    const auto in_height = input.size(3);
    const auto in_width = input.size(4);

    const auto out_channels = weight.size(0);
    const auto kernel_depth = weight.size(2);
    const auto kernel_height = weight.size(3);
    const auto kernel_width = weight.size(4);

    // Calculate output shape
    const auto out_depth = (in_depth - 1) * stride_d - 2 * padding_d + 
                          dilation_d * (kernel_depth - 1) + output_padding_d + 1;
    const auto out_height = (in_height - 1) * stride_h - 2 * padding_h + 
                           dilation_h * (kernel_height - 1) + output_padding_h + 1;
    const auto out_width = (in_width - 1) * stride_w - 2 * padding_w + 
                          dilation_w * (kernel_width - 1) + output_padding_w + 1;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, output_options);

    dim3 threads(kernel_depth, kernel_height, kernel_width);
    dim3 blocks(batch_size, out_channels, 1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose_3d_cuda", ([&] {
        conv_transpose_3d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            out_depth, out_height, out_width,
            in_depth, in_height, in_width,
            kernel_depth, kernel_height, kernel_width,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w,
            dilation_d, dilation_h, dilation_w,
            groups);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose_3d_cpp_source = """
torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight,
                                    int stride_d, int stride_h, int stride_w,
                                    int padding_d, int padding_h, int padding_w,
                                    int output_padding_d, int output_padding_h, int output_padding_w,
                                    int dilation_d, int dilation_h, int dilation_w,
                                    int groups);
"""

# Compile the custom CUDA kernel
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose_3d_cpp_source,
    cuda_sources=conv_transpose_3d_source,
    functions=["conv_transpose_3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the transposed convolution parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size, kernel_size)
        self.stride = (stride, stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights using same method as PyTorch
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Extract parameters
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        output_padding_d, output_padding_h, output_padding_w = self.output_padding
        dilation_d, dilation_h, dilation_w = self.dilation

        return conv_transpose3d.conv_transpose_3d_cuda(
            x, self.weight,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w,
            dilation_d, dilation_h, dilation_w,
            self.groups
        )