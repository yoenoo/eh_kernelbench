import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Define the kernel function for transposed convolution
template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4> input,
    const torch::PackedTensorAccessor<scalar_t,4> weight,
    torch::PackedTensorAccessor<scalar_t,4> output,
    int batch_size, int in_channels, int out_channels,
    int kH, int kW, int outH, int outW, int strideH, int strideW,
    int paddingH, int paddingW) {

    const int B = blockIdx.z;
    const int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int x_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (y_out >= outH || x_out >= outW) return;

    for (int c_out = threadIdx.z; c_out < out_channels; c_out += blockDim.z) {
        scalar_t sum = 0;
        for (int kiy = 0; kiy < kH; ++kiy) {
            for (int kix = 0; kix < kW; ++kix) {
                const int y_in = y_out + paddingH - kiy * strideH;
                const int x_in = x_out + paddingW - kix * strideW;
                if (y_in >= 0 && y_in < input.size(2) && x_in >=0 && x_in < input.size(3)) {
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        sum += weight[c_out][c_in][kiy][kix] * input[B][c_in][y_in][x_in];
                    }
                }
            }
        }
        output[B][c_out][y_out][x_out] = sum;
    }
}

// Wrapper function to call the kernel
at::Tensor conv_transpose2d_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    int strideH, int strideW,
    int paddingH, int paddingW) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0);
    const auto kH = weight.size(2);
    const auto kW = weight.size(3);

    // Compute output dimensions based on input and parameters
    const int inputH = input.size(2);
    const int inputW = input.size(3);
    const int outH = (inputH - 1) * strideH - 2 * paddingH + kH;
    const int outW = (inputW - 1) * strideW - 2 * paddingW + kW;

    auto output = at::zeros({batch_size, out_channels, outH, outW}, input.options());

    // Define grid and block dimensions
    dim3 threads(16, 16, 8); // Z dimension for channels
    dim3 blocks(
        (outW + threads.x - 1) / threads.x,
        (outH + threads.y - 1) / threads.y,
        batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            batch_size, in_channels, out_channels,
            kH, kW, outH, outW,
            strideH, strideW,
            paddingH, paddingW);
    }));

    return output;
}
"""

conv_transpose2d_cpp_source = "at::Tensor conv_transpose2d_cuda(const at::Tensor& input, const at::Tensor& weight, int strideH, int strideW, int paddingH, int paddingW);"

# Compile the CUDA kernel
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize parameters similar to PyTorch's ConvTranspose2d
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))  # Kernel weights

    def forward(self, x):
        return conv_transpose2d.conv_transpose2d_cuda(
            x, self.weight, self.stride[0], self.stride[1], self.padding[0], self.padding[1]
        )