import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void conv_transpose2d_kernel(const T* input, const T* weight, T* output,
    int batch_size, int in_channels, int out_channels, int kernel_size,
    int input_height, int input_width, int output_height, int output_width,
    int stride, int padding, int output_padding) {

    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * out_channels * output_height * output_width) return;

    int w_out = output_idx % output_width;
    int h_out = (output_idx / output_width) % output_height;
    int c_out = (output_idx / (output_width * output_height)) % out_channels;
    int n = output_idx / (out_channels * output_width * output_height);

    T val = 0;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            // Compute corresponding input position
            int h_in = (h_out + padding - kh) / stride;
            int w_in = (w_out + padding - kw) / stride;

            // Check if this position contributes to input
            if ((h_out + padding - kh) % stride == 0 && (w_out + padding - kw) % stride == 0 &&
                h_in >= -output_padding && h_in < input_height &&
                w_in >= 0 && w_in < input_width) {

                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    // Compute weight index
                    const int weight_idx = c_in * out_channels * kernel_size * kernel_size +
                                          c_out * kernel_size * kernel_size +
                                          kh * kernel_size + kw;
                    T w = weight[weight_idx];

                    // Compute input index
                    const int input_idx = n * in_channels * input_height * input_width +
                                         c_in * input_height * input_width +
                                         h_in * input_width + w_in;
                    val += input[input_idx] * w;
                }
            }
        }
    }

    // Compute output storage location
    int output_offset = n * out_channels * output_height * output_width +
                       c_out * output_height * output_width +
                       h_out * output_width + w_out;
    output[output_offset] = val;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight,
    int stride, int padding, int output_padding, int kernel_size) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(1) * weight.size(0); // Assuming weight dimensions [in_channels, out_channels_per_in, kernel, kernel]

    // Compute output dimensions using PyTorch's formula
    const int output_height = (input_height - 1) * stride - 2 * padding + 
                            kernel_size + output_padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + 
                            kernel_size + output_padding;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, 
        torch::dtype(input.dtype()).device(input.device()));

    dim3 threads(256);
    dim3 blocks((batch_size * out_channels * output_height * output_width + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, out_channels, kernel_size,
            input_height, input_width, output_height, output_width,
            stride, padding, output_padding);
    }));

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight,
    int stride, int padding, int output_padding, int kernel_size);
"""

conv_transpose2d_op = load_inline(
    name="conv_transpose2d_op",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weight with same initialization as PyTorch's ConvTranspose2d
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert weight to correct shape [in_channels * out_channels_per_in * kernel * kernel]
        # The current kernel expects weight to be arranged as [in_channels, out_channels_per_in, kernel, kernel]
        # which matches the PyTorch's ConvTranspose2d's weight layout
        weight = self.weight.view(self.weight.size(0), self.weight.size(1), -1)
        return conv_transpose2d_op.conv_transpose2d_cuda(
            x.contiguous(), 
            self.weight.contiguous(),
            self.stride, self.padding, self.output_padding, weight.size(-1)
        )