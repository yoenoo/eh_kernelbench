import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D convolution
conv1d_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void custom_conv1d_kernel(const scalar_t* __restrict__ input,
                                    const scalar_t* __restrict__ weight,
                                    scalar_t* __restrict__ output,
                                    int batch_size,
                                    int in_channels,
                                    int out_channels,
                                    int input_length,
                                    int kernel_size,
                                    int stride,
                                    int dilation,
                                    int output_length) {
    const int batch_idx = blockIdx.x;
    const int out_channel = blockIdx.y;
    const int time_idx = blockIdx.z * blockDim.x + threadIdx.x;

    if (time_idx >= output_length) return;

    scalar_t sum = 0;
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int k = 0; k < kernel_size; ++k) {
            const int input_time = time_idx * stride + k * dilation;
            if (input_time < input_length) {
                sum += input[batch_idx * in_channels * input_length + in_ch * input_length + input_time] *
                    weight[out_channel * in_channels * kernel_size + in_ch * kernel_size + k];
            }
        }
    }

    output[batch_idx * out_channels * output_length + out_channel * output_length + time_idx] = sum;
}

torch::Tensor custom_conv1d(torch::Tensor input, torch::Tensor weight, int stride, int dilation, int kernel_size) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int out_channels = weight.size(0);
    const int output_length = (input_length - kernel_size * dilation - stride + stride) / stride;

    auto output = torch::empty({batch_size, out_channels, output_length}, 
                             input.options());

    int block_size = 256;
    dim3 blocks(batch_size, out_channels, (output_length + block_size - 1) / block_size);
    dim3 threads(block_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv1d_cuda", ([&]{
        custom_conv1d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_length,
            kernel_size,
            stride,
            dilation,
            output_length);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv1d_kernel_header = """
torch::Tensor custom_conv1d(torch::Tensor input, torch::Tensor weight, int stride, int dilation, int kernel_size);
"""

# Compile the custom CUDA kernel
custom_conv_module = load_inline(
    name="custom_conv",
    cpp_sources=conv1d_kernel_header,
    cuda_sources=conv1d_kernel_source,
    functions=["custom_conv1d"],
    verbose=True,
    extra_cflags=["-D_FORCE_INLINES"],
    extra_cuda_cflags=["--expt-extended-lambda"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        # Initialize weights manually to avoid PyTorch's default initialization
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        # Bind the custom kernel
        self.custom_conv = custom_conv_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.custom_conv.custom_conv1d(x.cuda(), self.weight.cuda(), self.stride, self.dilation, self.kernel_size)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1).cuda()
        return output