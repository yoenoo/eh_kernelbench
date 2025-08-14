import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for 2D average pooling
avg_pool_2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool_2d_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
                                  torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
                                  int kernel_size, int stride, int padding) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    const int output_height = output.size(2);
    const int output_width = output.size(3);

    CUDA_KERNEL_LOOP(index, output.numel()) {
        int n = index / (channels * output_height * output_width);
        int c = (index / (output_height * output_width)) % channels;
        int oh = (index / output_width) % output_height;
        int ow = index % output_width;

        // Compute input coordinates with padding
        int h_start = oh * stride - padding;
        int w_start = ow * stride - padding;
        int h_end = h_start + kernel_size;
        int w_end = w_start + kernel_size;

        scalar_t sum = 0;
        int count = 0;

        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                if (h >= 0 && h < input_height && w >= 0 && w < input_width) {
                    sum += input[n][c][h][w];
                    count++;
                }
            }
        }

        output[n][c][oh][ow] = sum / count;
    }
}

std::vector<torch::Tensor> avg_pool_2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    // Calculate output dimensions
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, channels, output_height, output_width}, input.options());

    dim3 threads(256);
    dim3 blocks((output.numel() + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool_2d_cuda", ([&] {
        avg_pool_2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            kernel_size, stride, padding);
    }));

    return {output};
}
"""

avg_pool_2d_cpp_source = R"(
std::vector<torch::Tensor> avg_pool_2d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding
);
)"

# Compile the inline CUDA code for 2D average pooling
avg_pool_2d = load_inline(
    name="avg_pool_2d",
    cpp_sources=[avg_pool_2d_cpp_source],
    cuda_sources=[avg_pool_2d_source],
    functions=["avg_pool_2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        return avg_pool_2d.avg_pool_2d_cuda(x, self.kernel_size, self.stride, self.padding)[0]