import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose_source = """
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w) {
    int n = blockIdx.z;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (h_out >= output.size(2) || w_out >= output.size(3)) {
        return;
    }

    scalar_t val = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_in = h_out * stride_h - padding_h + kh;
            int w_in = w_out * stride_w - padding_w + kw;

            if (h_in >= 0 && h_in < input.size(2) && w_in >= 0 && w_in < input.size(3)) {
                for (int i_c = 0; i_c < input.size(1); ++i_c) {
                    val += input[n][i_c][h_in][w_in] * weight[i_c][n_c][kh][kw];
                }
            }
        }
    }
    output[n][n_c][h_out][w_out] = val;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    int output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_h;
    int output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_w;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    int block_dim_x = 32;
    int block_dim_y = 8;
    dim3 block(block_dim_x, block_dim_y, 1);

    dim3 grid(
        (output_width + block_dim_x -1)/block_dim_x,
        (output_height + block_dim_y -1)/block_dim_y,
        batch_size * out_channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<grid, block>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
    }));

    return output;
}
"""

conv_transpose_cpp_source = "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w);"

conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=['-gencode=arch=compute_86,code=sm_86'],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), bias=False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = conv_transpose.conv_transpose2d_cuda(
            x,
            self.weight,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1]
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output