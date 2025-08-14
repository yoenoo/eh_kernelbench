import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import numpy as np

# Define the fused Conv3D + ReLU custom CUDA kernel
conv3d_relu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel for Conv3D + ReLU
template <typename scalar_t>
__global__ void conv3d_relu_kernel(
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> output,
    int Cin, int Cout, int K, int S, int P, int D) {

    const int batch = blockIdx.z;
    const int out_depth = blockIdx.y;
    const int out_height = blockIdx.x;
    const int out_width = threadIdx.z;
    const int out_channel = threadIdx.y * 32 + threadIdx.x; // 2D thread arrangement

    if (out_channel >= Cout) return;

    scalar_t sum = 0;
    for (int in_channel = 0; in_channel < Cin; ++in_channel) {
        for (int d = 0; d < K; ++d) {
            for (int h = 0; h < K; ++h) {
                for (int w = 0; w < K; ++w) {
                    // Compute input spatial indices with padding/dilation
                    int in_d = out_depth * S + d * D - P;
                    int in_h = out_height * S + h * D - P;
                    int in_w = out_w * S + w * D - P;
                    
                    // Check input bounds
                    if (in_d < 0 || in_d >= input.size(2) || in_h < 0 || in_h >= input.size(3) || in_w < 0 || in_w >= input.size(4)) {
                        continue;
                    }

                    sum += input[batch][in_channel][in_d][in_h][in_w] * weight[out_channel][in_channel][d][h][w];
                }
            }
        }
    }

    // Apply ReLU activation
    output[batch][out_channel][out_depth][out_height][out_width] = sum > 0 ? sum : 0;
}

// Forward function
torch::Tensor conv3d_relu_forward(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation) {
    // Get dimensions
    int B = input.size(0);
    int Cin = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    int Cout = weight.size(0);
    int K = weight.size(2); // Assume square kernel (KxKxK)
    
    int out_depth = (D + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    int out_height = (H + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    int out_width = (W + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    // Initialize output tensor
    auto output = torch::zeros({B, Cout, out_depth, out_height, out_width}, input.options());

    // Define grid and block dimensions
    dim3 threads(32, 16, 1); // X: 32, Y: 16 (split output channels), Z: 1
    dim3 blocks(out_width, out_height, out_depth, B);

    // Launch kernel
    conv3d_relu_kernel<<<blocks, threads>>>(
        input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
        weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
        output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
        Cin, Cout, K, stride, padding, dilation
    );

    return output;
}
"""

conv3d_relu_cpp = "torch::Tensor conv3d_relu_forward(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"

# Load the custom kernel
conv3d_relu = load_inline(
    name="conv3d_relu",
    cpp_sources=conv3d_cpp,
    cuda_sources=conv3d_relu_source,
    functions=["conv3d_relu_forward"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-lineinfo"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv3d_relu.conv3d_relu_forward(
            x, self.weight, self.stride, self.padding, self.dilation
        )