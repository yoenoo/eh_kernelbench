import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Average Pooling
average_pool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void average_pool2d_kernel(const torch::PackedTensorAccessor<scalar_t,4> input,
                                     torch::PackedTensorAccessor<scalar_t,4> output,
                                     int kernel_size, int stride, int padding) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int output_height = output.size(2);
    const int output_width = output.size(3);

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_height * output_width) {
        return;
    }

    const int b = idx / (channels * output_height * output_width);
    const int c = (idx / (output_height * output_width)) % channels;
    const int oh = (idx / output_width) % output_height;
    const int ow = idx % output_width;

    // Compute input coordinates with padding
    const int h_start = oh * stride - padding;
    const int w_start = ow * stride - padding;
    const int h_end = h_start + kernel_size;
    const int w_end = w_start + kernel_size;

    scalar_t sum = 0.0;
    int count = 0;

    for (int h = h_start; h < h_end; ++h) {
        for (int w = w_start; w < w_end; ++w) {
            if (h >= 0 && h < input_height && w >= 0 && w < input_width) {
                sum += input[b][c][h][w];
                count++;
            }
        }
    }

    if (count > 0) {
        output[b][c][oh][ow] = sum / count;
    }
}

std::tuple<torch::Tensor, int, int, int> average_pool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    // Compute output dimensions
    const int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({batch_size, channels, output_height, output_width}, options);

    int total_elements = batch_size * channels * output_height * output_width;
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "average_pool2d_cuda", ([&] {
        average_pool2d_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            kernel_size,
            stride,
            padding
        );
    }));

    return std::make_tuple(output, output_height, output_width, count);
}

PYTHON체계 관리자(관리자계정) void average_pool2d(torch::Tensor input, int kernel_size, int stride, int padding) {
    return average_pool2d_cuda(input, kernel_size, stride, padding);
}
"""

average_pool2d_cpp_source = """
std::tuple<torch::Tensor, int, int, int> average_pool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
"""

# Compile the custom CUDA kernel
average_pool2d = load_inline(
    name="average_pool2d",
    cpp_sources=average_pool2d_cpp_source,
    cuda_sources=average_pool2d_source,
    functions=["average_pool2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _, _, _ = average_pool2d.average_pool2d_cuda(
            x, self.kernel_size, self.stride, self.padding
        )
        return output

def get_inputs():
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size]