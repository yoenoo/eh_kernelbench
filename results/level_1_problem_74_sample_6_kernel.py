import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

# CUDA kernel for optimized transposed 1D convolution
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv1d_transpose_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> output,
    const int in_channels, const int out_channels, const int kernel_size,
    const int stride, const int padding, const int dilation) {
    // Implement optimized transposed conv logic here
    // Consider parallelization across output channels and spatial dimensions
    // Calculate the input and output dimensions
    const int batch_size = input.size(0);
    const int input_length = input.size(2);
    const int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + 2 * padding;
    
    // Calculate dilation effect
    const int effective_kernel_size = (kernel_size - 1) * dilation + 1;
    const int output_padding = output_length - (input_length - 1) * stride + 2 * padding - effective_kernel_size + 1;
    
    // Iterate over output tensor elements in parallel
    int batch_idx = blockIdx.x;
    int out_channel = blockIdx.y;
    int out_pos = threadIdx.x + blockDim.x * blockIdx.z;
    
    if (out_pos >= output_length) return;
    
    for (int in_channel = 0; in_channel < in_channels; in_channel++) {
        // Calculate input position considering stride and dilation
        for (int k = 0; k < kernel_size; ++k) {
            int dilated_k = k * dilation;
            int input_pos = (out_pos - padding + dilated_k - output_padding) / stride;
            if ((out_pos - padding + dilated_k - output_padding) % stride == 0 &&
                input_pos >= 0 && input_pos < input_length) {
                atomicAdd(&output[batch_idx][out_channel][out_pos],
                         input[batch_idx][in_channel][input_pos] * weight[in_channel][out_channel][k]);
            }
        }
    }
}

at::Tensor conv1d_transpose_cuda(
    const at::Tensor input,
    const at::Tensor weight,
    int stride, int padding, int dilation) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int output_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    const int input_length = input.size(2);
    
    // Calculate output length
    const int effective_kernel_size = (kernel_size - 1) * dilation + 1;
    const int output_length = (input_length - 1) * stride - 2 * padding + effective_kernel_size;
    
    auto output = at::empty({batch_size, output_channels, output_length}, input.options());
    
    dim3 threads(256);
    dim3 blocks(batch_size, output_channels, 
               (output_length + threads.x - 1) / threads.x);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv1d_transpose_cuda", ([&] {
        conv1d_transpose_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>(),
            in_channels, output_channels, kernel_size,
            stride, padding, dilation);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_transpose", &conv1d_transpose_cuda, "Custom transposed 1D convolution");
}
"""

# Compile the CUDA extension
current_dir = os.path.dirname(os.path.abspath(__file__))
conv_transpose_cuda = load(name="conv_transpose_cuda", sources=[current_dir + "/conv_transpose_cuda.cpp"],
                           extra_cuda_cflags=['-gencode=arch=compute_80,code=sm_80'],
                           extra_cflags=['-O3'])

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weight same as PyTorch's ConvTranspose1d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        output = conv_transpose_cuda.conv1d_transpose(
            x, self.weight, self.stride, self.padding, self.dilation)
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
        
        return output