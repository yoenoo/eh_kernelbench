import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L1 normalization
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

template <typename T>
__global__ void l1_norm_kernel(const T* x, T* out, int batch_size, int dim) {
    extern __shared__ char _shared_space[];
    T* shared = reinterpret_cast<T*>(_shared_space);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    T sum = 0.0;
    if (idx < batch_size * dim) {
        sum += torch::abs(x[idx]);
    }

    // Compute the block-wise sum using shared memory
    __shared__ T block_sums[256];
    block_sums[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_sums[tid] += block_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        shared[blockIdx.x] = block_sums[0];
    }
    __syncthreads();

    // Now gather the block sums across all blocks using CUB
    if (threadIdx.x == 0) {
        T total_sum = 0.0;
        for (int i = 0; i < gridDim.x; ++i) {
            total_sum += shared[i * blockDim.x + 0];
        }
        shared[blockIdx.x] = total_sum;
    }
    __syncthreads();

    if (idx < batch_size * dim) {
        T total_sum = shared[blockIdx.x];
        if (total_sum != 0) {
            out[idx] = x[idx] / total_sum;
        } else {
            out[idx] = 0;
        }
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    auto out = torch::empty_like(x);
    
    const int threads = 256;
    const int blocks = (batch_size * dim + threads - 1) / threads;

    // Allocate shared memory per block for partial sums and block_sums array
    int shared_size = 256 * sizeof(float);
    l1_norm_kernel<float><<<blocks, threads, shared_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        dim
    );

    return out;
}
"""

l1_norm_cpp_source = (
    "torch::Tensor l1_norm_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for L1 normalization
l1_norm = load_inline(
    name="l1_norm",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_norm_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["--expt-relaxed-constexpr"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_norm = l1_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_norm.l1_norm_cuda(x)