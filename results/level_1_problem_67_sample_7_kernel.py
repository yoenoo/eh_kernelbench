import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 1D convolution
conv1d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv1d_kernel(const scalar_t* __restrict__ input,
                             const scalar_t* __restrict__ weight,
                             scalar_t* __restrict__ output,
                             int batch_size, int in_channels, int out_channels,
                             int input_length, int kernel_size, int output_length,
                             int stride, int padding, int dilation) {
    int B = blockIdx.x;  // Batch index
    int oC = blockIdx.y; // Output channel
    int p = threadIdx.x; // Position in the output feature map

    // Compute input position and validity
    int input_pos = p * stride - padding;
    if (input_pos < 0 || input_pos >= input_length) return;

    // Accumulate the convolution result
    scalar_t acc = 0;
    for (int k = 0; k < kernel_size; ++k) {
        int w_pos = k;  // Position in the kernel
        int in_pos = input_pos + dilation * k;
        if (in_pos < 0 || in_pos >= input_length) continue;
        for (int iC = 0; iC < in_channels; ++iC) {
            acc += weight[oC * in_channels * kernel_size + iC * kernel_size + k] *
                   input[B * in_channels * input_length + iC * input_length + in_pos];
        }
    }
    output[B * out_channels * output_length + oC * output_length + p] = acc;
}

std::tuple<torch::Tensor, int, int> conv1d_cuda(torch::Tensor input,
                                               torch::Tensor weight,
                                               int stride,
                                               int padding,
                                               int dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    // Calculate output dimensions
    const int output_length = (input_length + 2 * padding -
        dilation * (kernel_size - 1) - 1) / stride + 1;

    // Output tensor
    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    int blocks = batch_size * out_channels;
    int threads = output_length;

    // Launch kernel
    conv1d_kernel<scalar_t>
        <<<blocks, threads>>>(input.data_ptr<scalar_t>(),
                            weight.data_ptr<scalar_t>(),
                            output.data_ptr<scalar_t>(),
                            batch_size, in_channels, out_channels,
                            input_length, kernel_size, output_length,
                            stride, padding, dilation);

    return std::make_tuple(output, output_length, output_length);
}
"""

conv1d_cpp_source = """
std::tuple<torch::Tensor, int, int> conv1d_cuda(torch::Tensor input,
                                               torch::Tensor weight,
                                               int stride,
                                               int padding,
                                               int dilation);
"""

# Compile the custom CUDA code
conv1d_module = load_inline(
    name="conv1d_cuda",
    cpp_sources=conv1d_cpp_source,
    cuda_sources=conv1d_source,
    functions=["conv1d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups  # Currently not used in the kernel, may need adjustment for grouped convolutions
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _, _ = conv1d_module.conv1d_cuda(
            x.cuda(),
            self.weight.cuda(),
            self.stride,
            self.padding,
            self.dilation
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1).cuda()
        return output