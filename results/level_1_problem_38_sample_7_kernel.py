import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for L1 normalization
l1_norm_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256

template <typename scalar_t>
__global__ void l1_norm_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, int batch_size, int dim) {
    extern __shared__ unsigned char sdata[];
    scalar_t* s_sums = (scalar_t*)sdata;
    scalar_t* s_norm = s_sums + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    scalar_t sum = 0;

    if (idx < batch_size) {
        // Compute sum of absolute values for each sample
        for (int d = 0; d < dim; ++d) {
            scalar_t val = abs(x[idx * dim + d]);
            sum += val;
        }
    }

    // Write to shared memory
    s_sums[tid] = sum;
    __syncthreads();

    // Block reduction to compute total norm
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sums[tid] += s_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_norm[blockIdx.x] = s_sums[0];
    }
    __syncthreads();

    // Read norm and compute normalized values
    scalar_t norm = (tid < gridDim.x) ? s_norm[tid] : 0.0;
    __shared__ scalar_t block_norm;
    if (tid == 0) {
        block_norm = norm;
    }
    __syncthreads();

    norm = block_norm;

    if (idx < batch_size) {
        for (int d = 0; d < dim; ++d) {
            int pos = idx * dim + d;
            y[pos] = x[pos] / norm;
        }
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x) {
    const auto batch_size = x.size(0);
    const auto dim = x.size(1);
    const auto total_elements = batch_size * dim;
    const int block_size = THREADS_PER_BLOCK;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    auto y = torch::empty_like(x);
    
    // Define shared memory size
    size_t smem_size = (block_size + num_blocks) * sizeof(float);

    l1_norm_kernel<float><<<num_blocks, block_size, smem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size,
        dim
    );

    return y;
}
"""

# Compile the inline CUDA code
l1_norm_cpp_source = "torch::Tensor l1_norm_cuda(torch::Tensor x);"
l1_norm = load_inline(
    name="l1_norm",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l1_norm = l1_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_norm.l1_norm_cuda(x)