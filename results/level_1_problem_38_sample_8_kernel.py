import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L1 normalization
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void l1_norm_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, const int batch_size, const int dim) {
    extern __shared__ scalar_t shared_data[];
    int batch = blockIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    scalar_t sum = 0.0;
    for (int i = tid; i < dim; i += blockDim.x) {
        scalar_t val = abs(x[batch * dim + i]);
        sum += val;
        shared_data[i] = val; // Not used here, but structure preserved for potential extension
    }

    __syncthreads();

    // Compute the sum across threads
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sum += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write back the sum to global memory for each batch (here optimized for single block per batch)
    if (tid == 0) {
        shared_data[0] = sum;
    }
    __syncthreads();

    scalar_t inv_sum = 1.0 / shared_data[0];

    // Normalize and write results
    for (int i = tid; i < dim; i += blockDim.x) {
        y[batch * dim + i] = x[batch * dim + i] * inv_sum;
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);

    auto y = torch::empty_like(x);

    const int block_size = 256;
    dim3 blocks(batch_size);
    dim3 threads(block_size);
    size_t shared_size = dim * sizeof(float); // Adjust if needed for data type

    // Launch kernel with shared memory for intermediate storage
    l1_norm_kernel<float><<<blocks, threads, shared_size, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, dim);

    return y;
}
"""

l1_norm_cpp_source = (
    "torch::Tensor l1_norm_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code
l1_norm = load_inline(
    name="l1_norm",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_norm_cuda"],
    verbose=False,
    extra_cflags=["-gencode=arch=compute_75,code=sm_75"],  # Adjust for your GPU
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_norm = l1_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_norm.l1_norm_cuda(x)