import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Custom CUDA kernel for optimized 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void custom_conv3d_forward(
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> output,
    const int batch_size, const int in_channels, const int depth, const int width, const int height,
    const int out_channels, const int kernel_size, const int stride, const int padding, const int dilation) {
    
    const int depth_out = (depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int width_out = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int height_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    const int output_z = blockIdx.z;
    const int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int output_c = blockIdx.x * gridDim.x + blockIdx.y;

    if (output_x >= width_out || output_y >= height_out || output_c >= out_channels) return;

    const int in_channel_group = threadIdx.z;
    scalar_t sum = 0;

    for (int kernel_d = 0; kernel_d < kernel_size; ++kernel_d) {
        for (int kernel_h = 0; kernel_h < kernel_size; ++kernel_h) {
            for (int kernel_w = 0; kernel_w < kernel_size; ++kernel_w) {
                const int input_d = output_z * stride - padding + dilation * kernel_d;
                const int input_x = output_x * stride - padding + dilation * kernel_h;
                const int input_y = output_y * stride - padding + dilation * kernel_w;
                
                if (input_d >= 0 && input_d < depth &&
                    input_x >= 0 && input_x < width &&
                    input_y >= 0 && input_y < height) {
                    
                    for (int c = 0; c < in_channels; c += blockDim.z) { // Assuming in_channels is divisible by blockDim.z
                        int in_c = c + in_channel_group;
                        if (in_c < in_channels) {
                            sum += input[output_z][in_c][input_d][input_x][input_y] * 
                                   weight[output_c][in_c][kernel_d][kernel_h][kernel_w];
                        }
                    }
                }
            }
        }
    }

    output[output_z][output_c][output_x][output_y] = sum;
}

torch::Tensor custom_conv3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int depth = input.size(2);
    const int width = input.size(3);
    const int height = input.size(4);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    auto output = torch::zeros({batch_size, out_channels, 
        (depth + 2 * padding - dilation * (kernel_size -1 ) -1)/stride +1,
        (width + 2 * padding - dilation * (kernel_size -1 ) -1)/stride +1,
        (height + 2 * padding - dilation * (kernel_size -1 ) -1)/stride +1}, input.options());

    const dim3 threads(16, 16, 8); // Thread configuration for 3D blocks
    const dim3 blocks(2, 2, output.size(2)); // Block configuration adjusted based on output dimensions

    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv3d_forward", ([&] {
        custom_conv3d_forward<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            batch_size, in_channels, depth, width, height,
            out_channels, kernel_size, stride, padding, dilation);
    }));

    return output;
}
"""

conv3d_cpp_source = "torch::Tensor custom_conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"

# Compile the inline CUDA code for the convolution
custom_conv3d = load_inline(
    name="custom_conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["custom_conv3d_forward_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        
    def forward(self, x):
        return custom_conv3d.custom_conv3d_forward_cuda(x, self.weight, self.stride, self.padding, self.dilation)