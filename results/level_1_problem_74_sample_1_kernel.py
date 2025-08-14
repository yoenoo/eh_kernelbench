import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel implementation for ConvTranspose1d
conv_transpose_1d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_length,
    int output_length,
    int stride,
    int padding,
    int dilation
) {
    int B = blockIdx.z;
    int out_ch = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= output_length) return;

    scalar_t val = 0;
    for (int filt = 0; filt < kernel_size; ++filt) {
        int in_col = col + padding - dilation * filt;
        if (in_col < 0 || in_col >= input_length) continue;
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            val += weight[out_ch * in_channels * kernel_size + in_ch * kernel_size + filt] *
                   input[B * in_channels * input_length + in_ch * input_length + in_col];
        }
    }
    output[B * out_channels * output_length + out_ch * output_length + col] = val;
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
    const int input_length = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    int output_length = (input_length - 1) * stride - 2 * padding + 
        dilation * (kernel_size - 1) + 1;

    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    const int threads = 256;
    dim3 blocks(
        (output_length + threads - 1) / threads,
        out_channels,
        batch_size
    );

    const int shared_mem = 0;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose1d_cuda", ([&] {
        conv_transpose1d_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            in_channels,
            out_channels,
            kernel_size,
            input_length,
            output_length,
            stride,
            padding,
            dilation
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose_1d_header = """
torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
);
"""

conv_transposed = load_inline(
    name="conv_transposed",
    cpp_sources=conv_transpose_1d_header,
    cuda_sources=conv_transpose_1d_source,
    functions=["conv_transpose1d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights similar to PyTorch's ConvTranspose1d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = conv_transposed.conv_transpose1d_cuda(
            x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)
        return out