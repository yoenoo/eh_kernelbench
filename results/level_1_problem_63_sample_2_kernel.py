import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D convolution
conv2d_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void conv2d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
                                    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
                                    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
                                    int batch_size, int in_channels, int input_height, int input_width,
                                    int out_channels, int kernel_size, int stride, int padding, int dilation) {

    const int H_out = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int W_out = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;

    if (col >= W_out || row >= H_out || batch >= batch_size) return;

    for (int f = 0; f < out_channels; ++f) {
        float val = 0;
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int h_idx = row * stride + i * dilation - padding;
                int w_idx = col * stride + j * dilation - padding;
                if (h_idx >= 0 && h_idx < input_height && w_idx >= 0 && w_idx < input_width) {
                    for (int c = 0; c < in_channels; ++c) {
                        val += input[batch][c][h_idx][w_idx] * weight[f][c][i][j];
                    }
                }
            }
        }
        output[batch][f][row][col] = val;
    }
}

torch::Tensor conv2d_forward_cuda(torch::Tensor input, torch::Tensor weight,
                                int stride, int padding, int dilation) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    int block_x = 32;
    int block_y = 8;
    dim3 block(block_x, block_y);
    dim3 grid( (output_width + block_x - 1)/block_x, (output_height + block_y - 1)/block_y, batch_size );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_forward", ([&] {
        conv2d_forward_kernel<scalar_t><<<grid, block>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, in_channels, input_height, input_width,
            out_channels, kernel_size, stride, padding, dilation
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv2d_cpp_source = "torch::Tensor conv2d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"

# Compile the inline CUDA code for convolution
conv2d_forward = load_inline(
    name="conv2d_forward",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_kernel_source,
    functions=["conv2d_forward_cuda"],
    verbose=True,
    extra_cflags=["-DTRACE"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize convolution weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        
    def forward(self, x):
        return conv2d_forward.conv2d_forward_cuda(x, self.weight, self.stride, self.padding, self.dilation)