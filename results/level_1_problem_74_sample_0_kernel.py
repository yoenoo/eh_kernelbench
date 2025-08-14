import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose1d
conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel is a simplified version of the transpose convolution. It assumes
// that the kernel is symmetric and no padding. It may not handle all cases.
template <typename scalar_t>
__global__ void conv_transpose1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_length,
    int output_length,
    int stride,
    int dilation
) {
    const int output_size = batch_size * out_channels * output_length;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    const int batch = idx / (out_channels * output_length);
    const int out_ch = (idx / output_length) % out_channels;
    const int out_pos = idx % output_length;

    scalar_t val = 0;
    for (int kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
        const int dilated_kernel = kernel_idx * dilation;
        const int input_pos = out_pos + dilated_kernel - (dilation * (kernel_size - 1));
        if (input_pos < 0 || input_pos >= input_length) continue;
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            val += input[batch * in_channels * input_length + in_ch * input_length + input_pos] *
                   weight[out_ch * in_channels * kernel_size + in_ch * kernel_size + kernel_idx];
        }
    }
    output[idx] = val;
}

torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int input_length = input.size(2);
    const int output_length = (input_length - 1) * stride + 1 - 2 * padding + (kernel_size - 1) * dilation;

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    const int threads = 256;
    const int blocks = (output.numel() + threads - 1) / threads;

    const int block_size = 256;
    const int num_blocks = (output.numel() + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose1d_cuda", ([&] {
        conv_transpose1d_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            input_length,
            output_length,
            stride,
            dilation
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = """
torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
);
"""

conv_transpose1d_cuda = load_inline(
    name="conv_transpose1d_cuda",
    cpp_sources=cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose1d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        output = conv_transpose1d_cuda.conv_transpose1d_cuda(
            x, self.weight, self.stride, self.padding, self.dilation
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
        return output