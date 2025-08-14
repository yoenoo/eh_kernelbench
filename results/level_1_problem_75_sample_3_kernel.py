import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4> input,
    const torch::PackedTensorAccessor<scalar_t,4> weight,
    torch::PackedTensorAccessor<scalar_t,4> output,
    int in_channels, int out_channels, int groups,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int output_h, int output_w,
    int batch_size) {

    CUDA_KERNEL_LOOP(index, batch_size * output_h * output_w) {
        int w_out = index % output_w;
        int h_out = (index / output_w) % output_h;
        int n = index / (output_w * output_h);

        for (int og = 0; og < out_channels / groups; ++og) {
            int og_group = og + groups * (index / (groups * output_h * output_w));
            int ig_group = og_group / (out_channels / groups);

            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int h_in = (h_out - padding_h - kh * dilation_h) / stride_h;
                    int w_in = (w_out - padding_w - kw * dilation_w) / stride_w;

                    if (h_in < 0 || w_in < 0 || h_in >= input.size(2) || w_in >= input.size(3))
                        continue;

                    for (int ic = 0; ic < in_channels / groups; ++ic) {
                        output[n][og_group][h_out][w_out] +=
                            weight[og_group][ic + ig_group * (in_channels / groups)][kh][kw] *
                            input[n][ic + ig_group * (in_channels / groups)][h_in][w_in];
                    }
                }
            }
        }
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups) {

    // Calculate output dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_h = input.size(2);
    auto input_w = input.size(3);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    auto out_channels = weight.size(0) * groups;

    auto output_h = (input_h - 1) * stride_h - 2 * padding_h +
                    dilation_h * (kernel_h - 1) + 1;
    auto output_w = (input_w - 1) * stride_w - 2 * padding_w +
                    dilation_w * (kernel_w - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());

    const int threads = 512;
    int elements = batch_size * output_h * output_w;
    int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda", ([&]{
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            in_channels, out_channels, groups,
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            output_h, output_w,
            batch_size);
    }));

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups);
"""

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias similar to PyTorch's ConvTranspose2d
        kh, kw = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kh, kw))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        output = conv_transpose2d.conv_transpose2d_cuda(
            x,
            self.weight,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output