import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 1D convolution
conv_transpose_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose_1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_length,
    int output_length,
    int stride,
    int padding,
    int dilation) {

    int batch = blockIdx.x;
    int out_channel = blockIdx.y;
    int out_pos = blockIdx.z * blockDim.x + threadIdx.x;

    if (out_pos >= output_length) return;

    const scalar_t* w_ptr = weight + out_channel * in_channels * kernel_size;
    const scalar_t* in_batch = input + batch * in_channels * input_length;
    scalar_t* out_batch = output + batch * out_channels * output_length;

    scalar_t val = 0;
    for (int kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
        int in_channel;
        int eff_kernel_idx = kernel_idx;
        int dilated_kernel_pos = eff_kernel_idx * dilation;

        // Determine corresponding input position
        int input_pos = (out_pos - dilated_kernel_pos - padding) / stride;
        if ((out_pos - dilated_kernel_pos - padding) % stride != 0)
            continue;
        if (input_pos < 0 || input_pos >= input_length)
            continue;

        for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
            scalar_t w = w_ptr[in_channel * kernel_size + kernel_idx];
            scalar_t in_val = in_batch[in_channel * input_length + input_pos];
            val += w * in_val;
        }
    }
    out_batch[out_channel * output_length + out_pos] = val;
}

torch::Tensor conv_transpose_1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int input_length = input.size(2);

    // Compute output length
    const int output_length = (input_length - 1) * stride - 2 * padding +
        dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    const int threads = 256;
    dim3 blocks(batch_size, out_channels, (output_length + threads - 1) / threads);

    const int block_size = 256;
    const int num_blocks = (output_length + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose_1d_cuda", ([&] {
        conv_transpose_1d_kernel<scalar_t><<<blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
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

    return output;
}
"""

conv_transpose_1d_cpp_source = (
    "torch::Tensor conv_transpose_1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"
)

# Compile the inline CUDA code
conv_transpose_1d = load_inline(
    name="conv_transpose_1d",
    cpp_sources=conv_transpose_1d_cpp_source,
    cuda_sources=conv_transpose_1d_source,
    functions=["conv_transpose_1d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        self.cuda_op = conv_transpose_1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cuda_op.conv_transpose_1d_cuda(x.cuda(), self.weight.cuda(), self.stride, self.padding, self.dilation)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1).cuda()
        return out