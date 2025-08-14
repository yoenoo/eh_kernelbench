import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 1D Average Pooling
avg_pool_1d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool_1d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_length,
    const int out_length,
    const int kernel_size,
    const int padding,
    const int stride
) {
    const int batch_id = blockIdx.x;
    const int channel = blockIdx.y;
    const int out_pos = blockIdx.z;
    const int in_start = out_pos * stride - padding;
    const int in_end = in_start + kernel_size;
    const int in_dim = in_length;
    scalar_t sum = 0;
    for (int i = in_start; i < in_end; ++i) {
        if (i >=0 && i < in_dim) {
            sum += input[batch_id * channels * in_dim + channel * in_dim + i];
        }
    }
    const int output_idx = batch_id * channels * out_length + channel * out_length + out_pos;
    output[output_idx] = sum / kernel_size;
}

std::tuple<torch::Tensor> avg_pool_1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_length = input.size(2);
    const int out_length = (in_length + 2 * padding - kernel_size) / stride + 1;

    torch::Tensor output = torch::empty({batch_size, channels, out_length}, input.options());

    dim3 blocks(batch_size, channels, out_length);
    dim3 threads(1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "avg_pool_1d_cuda", ([&]{
        avg_pool_1d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            channels,
            in_length,
            out_length,
            kernel_size,
            padding,
            stride
        );
    }));

    return output;
}
"""

avg_pool_1d_cpp_source = "std::tuple<torch::Tensor> avg_pool_1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"

# Compile the CUDA kernel
avg_pool_1d = load_inline(
    name='avg_pool_1d',
    cpp_sources=avg_pool_1d_cpp_source,
    cuda_sources=avg_pool_1d_source,
    functions=['avg_pool_1d_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool_cuda = avg_pool_1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool_cuda.avg_pool_1d_cuda(x, self.kernel_size, self.stride, self.padding)[0]