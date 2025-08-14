import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for L2 normalization
l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>

template <typename scalar_t>
__device__ scalar_t warp_reduce(scalar_t val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, mask);
    return val;
}

template <typename scalar_t>
__global__ void l2_norm_kernel(const scalar_t* __restrict__ input,
                              scalar_t* __restrict__ output,
                              int batch_size,
                              int dim) {
    extern __shared__ scalar_t norm_partial[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int global_idx = batch_idx * dim + tid;

    // Load data into shared memory
    scalar_t value = input[global_idx];

    // Each thread computes squared value
    norm_partial[tid] = value * value;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            norm_partial[tid] += norm_partial[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction for last 32 threads
    scalar_t sum_squares = warp_reduce<scalar_t>(norm_partial[tid]);
    if (tid == 0) {
        norm_partial[0] = sqrt(sum_squares) + 1e-12; // avoid division by zero
    }
    __syncthreads();

    // Apply normalization
    scalar_t inv_norm = 1.0 / norm_partial[0];
    output[global_idx] = value * inv_norm;
}

torch::Tensor l2_norm_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int dim = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 256;
    const dim3 blocks(batch_size);
    // Shared memory needed: threads per block for partial sums
    int smem_size = threads * sizeof(float);

    l2_norm_kernel<float><<<blocks, threads, smem_size, torch::cuda::current_stream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );

    return output;
}
"""

l2_norm_cpp_source = (
    "torch::Tensor l2_norm_cuda(torch::Tensor input);"
)

# Compile the custom CUDA operator
l2_norm = load_inline(
    name="l2_norm",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_norm_cuda"],
    verbose=True,
    extra_cflags=["-DDEBUG"],
    extra_cuda_cflags=["-DDEBUG"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_norm = l2_norm  # Holds the CUDA extension module

    def forward(self, x):
        return self.l2_norm.l2_norm_cuda(x)