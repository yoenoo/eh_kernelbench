import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for LogSoftmax
log_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void log_softmax_forward_kernel(const scalar_t* input, scalar_t* output,
                                          int batch_size, int dim_size, int dim) {
    const int batch_stride = dim_size;
    const int x_idx = blockIdx.x * batch_stride + threadIdx.x;

    // Compute row-wise max for numerical stability
    scalar_t row_max = -std::numeric_limits<scalar_t>::infinity();
    if (threadIdx.x < dim_size) {
        scalar_t val = input[x_idx];
        row_max = (val > row_max) ? val : row_max;
    }
    __shared__ scalar_t shared_max;
    if (threadIdx.x == 0) {
        shared_max = row_max;
    }
    __syncthreads();

    scalar_t max_val = shared_max;
    scalar_t sum_exp = 0;
    if (threadIdx.x < dim_size) {
        scalar_t exp_val = exp(input[x_idx] - max_val);
        sum_exp += exp_val;
    }
    __shared__ scalar_t shared_sum;
    if (threadIdx.x == 0) {
        shared_sum = sum_exp;
    }
    __syncthreads();

    if (threadIdx.x < dim_size) {
        output[x_idx] = log(shared_sum) + max_val;
        output[x_idx] = input[x_idx] - output[x_idx];
    }
}

at::Tensor log_softmax_forward_cuda(const at::Tensor& input, int64_t dim) {
    const auto batch_size = input.size(0);
    const auto dim_size = input.size(dim);
    auto output = at::empty_like(input);

    dim3 block_size(dim_size);
    dim3 grid_size(batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "log_softmax_forward", ([&] {
        log_softmax_forward_kernel<scalar_t><<<grid_size, block_size>>>(
            input.contiguous().data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            dim_size,
            dim);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

log_softmax_cpp_source = """
at::Tensor log_softmax_forward_cuda(const at::Tensor& input, int64_t dim);
"""

# Compile the inline CUDA code
log_softmax_module = load_inline(
    name="log_softmax",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=["log_softmax_forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.log_softmax_forward_cuda = log_softmax_module.log_softmax_forward_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_softmax_forward_cuda(x, self.dim)