import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define custom Conv1D CUDA kernel
conv1d_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

template <typename scalar_t>
__global__ void custom_conv1d_forward(const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
                                      const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> weight,
                                      torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output,
                                      int in_channels,
                                      int out_channels,
                                      int kernel_size,
                                      int stride,
                                      int padding,
                                      int dilation) {

    const int batch_idx = blockIdx.z;
    const int out_channel = blockIdx.y;
    const int in_channel = blockIdx.x % in_channels;
    const int thread_idx = threadIdx.x;

    const int output_length = output.size(1);
    const int input_length = input.size(1);

    __shared__ scalar_t shared_input[32 * 1024]; // Shared memory buffer for input block

    scalar_t acc = 0.0;

    for (int output_pos = blockIdx.x / in_channels * blockDim.x + thread_idx; output_pos < output_length; output_pos += blockDim.x * gridDim.x) {
        acc = 0.0;
        
        // Compute corresponding input position
        const int input_start = output_pos * stride - padding;
        const int input_end = input_start + dilation * (kernel_size - 1) + 1;

        // Load input segment into shared memory
        for (int k = 0; k < kernel_size; ++k) {
            int input_k = input_start + dilation * k;
            if (input_k >= 0 && input_k < input_length) {
                shared_input[thread_idx + k * blockDim.x] = input[in_channel][input_k];
            } else {
                shared_input[thread_idx + k * blockDim.x] = 0.0;
            }
        }
        __syncthreads();

        // Perform the convolution computation
        for (int k = 0; k < kernel_size; ++k) {
            acc += shared_input[thread_idx + k * blockDim.x] * weight[out_channel * in_channels * kernel_size + in_channel * kernel_size + k];
        }
        __syncthreads();
    }

    if (output_pos < output_length) {
        output[batch_idx][out_channel * output_length + output_pos] = acc;
    }
}

torch::Tensor custom_conv1d_forward_cuda(torch::Tensor input,
                                         torch::Tensor weight,
                                         int in_channels,
                                         int out_channels,
                                         int kernel_size,
                                         int stride,
                                         int padding,
                                         int dilation) {

    const int batch_size = input.size(0);
    const int input_length = input.size(2);
    const int output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    dim3 threads(256);
    dim3 blocks(
        (in_channels * ((output_length + threads.x - 1) / threads.x)),
        out_channels,
        batch_size
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv1d_forward", ([&] {
        custom_conv1d_forward<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            weight.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }));

    return output;
}
"""

conv1d_kernel_header = """
torch::Tensor custom_conv1d_forward_cuda(torch::Tensor input,
                                         torch::Tensor weight,
                                         int in_channels,
                                         int out_channels,
                                         int kernel_size,
                                         int stride,
                                         int padding,
                                         int dilation);
"""

# Compile the CUDA kernel
custom_conv1d = load_inline(
    name="custom_conv1d",
    cpp_sources=conv1d_kernel_header,
    cuda_sources=conv1d_kernel_source,
    functions=["custom_conv1d_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_conv1d.custom_conv1d_forward_cuda(
            x,
            self.weight,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        ).reshape(x.size(0), self.out_channels, -1)