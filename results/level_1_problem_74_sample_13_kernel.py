import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 1D convolution
conv1d_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void conv_transpose1d_kernel(scalar_t* __restrict__ out,
                                       const scalar_t* __restrict__ in,
                                       const scalar_t* __restrict__ weight,
                                       const scalar_t* __restrict__ bias,
                                       int batch_size, int in_channels, int out_channels,
                                       int input_length, int output_length,
                                       int kernel_size, int stride, int padding, int dilation, int dilation_out) {
    const int output_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;
    const int out_ch = blockIdx.z;

    if (output_pos >= output_length) return;

    scalar_t sum = bias[out_ch];
    for (int k = 0; k < kernel_size; ++k) {
        int in_pos = (output_pos - k * dilation_out - padding) / stride;
        if (in_pos < 0 || in_pos >= input_length) continue;
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            sum += in[batch * in_channels * input_length + in_ch * input_length + in_pos] *
                   weight[out_ch * in_channels * kernel_size + in_ch * kernel_size + k];
        }
    }
    out[batch * out_channels * output_length + out_ch * output_length + output_pos] = sum;
}

at::Tensor conv1d_transpose_cuda(const at::Tensor& input,
                                const at::Tensor& weight,
                                const at::Tensor& bias,
                                int stride,
                                int padding,
                                int dilation,
                                int dilation_out) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int output_length = (input_length - 1) * stride + (kernel_size - 1) * dilation_out + 1 - 2 * padding;

    auto output = at::empty({batch_size, out_channels, output_length}, input.type().cuda());

    const int threads = 256;
    const dim3 blocks(output_length, batch_size, out_channels);
    const int dilation_out_val = (dilation * stride); // Compute effective dilation for output

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv1d_transpose_cuda", ([&] {
        conv_transpose1d_kernel<scalar_t><<<blocks, threads>>>(
            output.data<scalar_t>(),
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            bias.data<scalar_t>(),
            batch_size, in_channels, out_channels,
            input_length, output_length,
            kernel_size, stride, padding, dilation, dilation_out_val);
    }));

    AT_CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

conv1d_transpose_cpp_source = """
at::Tensor conv1d_transpose_cuda(const at::Tensor& input,
                                const at::Tensor& weight,
                                const at::Tensor& bias,
                                int stride,
                                int padding,
                                int dilation,
                                int dilation_out);
"""

conv1d_transpose = load_inline(
    name="conv1d_transpose",
    cpp_sources=conv1d_transpose_cpp_source,
    cuda_sources=conv1d_transpose_source,
    functions=["conv1d_transpose_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize weights and bias similar to nn.ConvTranspose1d
        weight = torch.randn(out_channels, in_channels, kernel_size)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        self.weight = nn.Parameter(weight)
        self.conv1d_transpose = conv1d_transpose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1d_transpose.conv1d_transpose_cuda(
            x.cuda(),
            self.weight.cuda(),
            self.bias.cuda() if self.bias is not None else torch.zeros(self.out_channels).cuda(),
            self.stride,
            self.padding,
            self.dilation,
            self.dilation * self.stride
        )