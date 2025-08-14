import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 1D convolution
conv_transpose_1d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose_1d_kernel(const scalar_t* __restrict__ input,
                                        const scalar_t* __restrict__ weight,
                                        scalar_t* __restrict__ output,
                                        const int batch_size,
                                        const int in_channels,
                                        const int out_channels,
                                        const int kernel_size,
                                        const int output_length,
                                        const int input_length,
                                        const int stride,
                                        const int padding,
                                        const int dilation) {

    // Compute output position based on thread and block indices
    const int output_element = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_element >= batch_size * out_channels * output_length) {
        return;
    }

    const int output_channel = (output_element / output_length) % out_channels;
    const int output_pos = output_element % output_length;
    const int batch_idx = output_element / (out_channels * output_length);

    scalar_t sum = 0;
    for (int kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
        // Compute corresponding input position
        const int eff_kernel_size = (kernel_size - 1) * dilation + 1;
        const int reverse_kernel_idx = kernel_size - 1 - kernel_idx;
        const int input_pos = output_pos - reverse_kernel_idx * dilation;

        // Check if the input_pos is within bounds considering padding
        if (input_pos < -padding || input_pos >= input_length + padding) {
            continue;
        }

        // Map input_pos with padding into actual input indices
        const int padded_input_pos = input_pos + padding;
        const bool is_valid_input = padded_input_pos >= 0 && padded_input_pos < input_length;
        if (!is_valid_input) {
            continue;
        }

        for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
            // Get input value
            const int input_offset = batch_idx * in_channels * input_length +
                                     in_channel * input_length +
                                     padded_input_pos;
            const scalar_t in_val = input[input_offset];

            // Get weight value
            const int weight_offset = output_channel * in_channels * kernel_size +
                                      in_channel * kernel_size +
                                      reverse_kernel_idx;
            const scalar_t weight_val = weight[weight_offset];

            sum += in_val * weight_val;
        }
    }

    // Write to the output
    const int output_offset = batch_idx * out_channels * output_length +
                              output_channel * output_length +
                              output_pos;
    output[output_offset] = sum;
}

std::tuple<torch::Tensor, torch::Tensor> conv_transpose_1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    // Compute output length based on transposed conv formula
    const int output_length = (input_length - 1) * stride -
        2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    const int threads_per_block = 256;
    const int num_elements = batch_size * out_channels * output_length;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose_1d_cuda", ([&]{
        conv_transpose_1d_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            output_length,
            input_length,
            stride,
            padding,
            dilation);
    }));

    return std::make_tuple(output, weight); // Returning weight to avoid unused variable warning
}

"""
conv_transpose_1d_cpp_source = """
    std::tuple<torch::Tensor, torch::Tensor> conv_transpose_1d_cuda(
        torch::Tensor input,
        torch::Tensor weight,
        int stride,
        int padding,
        int dilation);
"""

conv_transpose_1d_op = load_inline(
    name="conv_transpose_1d_op",
    cpp_sources=conv_transpose_1d_cpp_source,
    cuda_sources=conv_transpose_1d_source,
    functions="conv_transpose_1d_cuda",
    verbose=True,
    extra_cflags=["-D_DEBUG"],
    extra_cuda_cflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: 1 = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # Weight has shape [out_channels, in_channels, kernel_size]
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        self.conv_transpose_1d_op = conv_transpose_1d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.conv_transpose_1d_op.conv_transpose_1d_cuda(
            x.cuda(),
            self.weight.cuda(),
            self.stride,
            self.padding,
            self.dilation
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1).cuda()
        return output