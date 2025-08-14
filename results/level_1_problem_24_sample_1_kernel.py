import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for LogSoftmax
log_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void log_softmax_forward_kernel(const scalar_t* __restrict__ input,
                                          scalar_t* __restrict__ output,
                                          int batch_size,
                                          int dim_size,
                                          int dim) {
    int batch_idx = blockIdx.x;
    int element_idx = threadIdx.x;

    if (element_idx >= dim_size)
        return;

    int index = batch_idx * dim_size + element_idx;
    scalar_t val = input[index];

    // Compute max for stability
    __shared__ scalar_t block_max;
    if (threadIdx.x == 0) {
        block_max = input[batch_idx * dim_size];
        for (int i = 1; i < dim_size; i++) {
            if (input[batch_idx * dim_size + i] > block_max) {
                block_max = input[batch_idx * dim_size + i];
            }
        }
    }
    __syncthreads();

    val -= block_max;
    scalar_t exp_val = exp(val);
    __shared__ scalar_t sum;
    if (threadIdx.x == 0)
        sum = 0;
    __syncthreads();
    atomicAdd(&sum, exp_val);
    __syncthreads();

    scalar_t log_sum = log(sum);
    output[index] = val - log_sum - block_max;
}

torch::Tensor log_softmax_forward_cuda(torch::Tensor input, int dim) {
    int batch_size = input.size(0);
    int dim_size = input.size(1);

    auto output = torch::empty_like(input);

    dim3 blocks(batch_size);
    dim3 threads(dim_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "log_softmax_forward_cuda", ([&] {
        log_softmax_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim_size,
            dim);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

log_softmax_cpp_source = """
torch::Tensor log_softmax_forward_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code for LogSoftmax
log_softmax = load_inline(
    name="log_softmax_cuda",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=["log_softmax_forward_cuda"],
    verbose=False,
    extra_cflags=["-DLOG_SOFTMAX_CUDA"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.log_softmax = log_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_softmax.log_softmax_forward_cuda(x, self.dim)