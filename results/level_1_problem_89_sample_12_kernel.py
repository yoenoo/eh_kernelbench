import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

scan_kernel = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void scan_kernel(const scalar_t* input, scalar_t* output, int dim_size, int batch_size, int other_dims) {
    extern __shared__ scalar_t shared_mem[];
    
    int batch_idx = blockIdx.z;
    int other_dim_idx = blockIdx.y;
    int thread_idx = threadIdx.x;
    int global_idx = batch_idx * other_dims * dim_size + other_dim_idx * dim_size + thread_idx;
    
    scalar_t val = (thread_idx < dim_size) ? input[global_idx] : 0;
    shared_mem[thread_idx] = val;
    __syncthreads();
    
    // Up-sweep phase
    for (int d = 1; d < dim_size; d <<= 1) {
        if (thread_idx >= d) {
            shared_mem[thread_idx] += shared_mem[thread_idx - d];
        }
        __syncthreads();
    }
    
    // Down-sweep phase
    for (int d = 1; d < dim_size; d <<= 1) {
        int o = d;
        d <<= 1;
        if (thread_idx < d) {
            if (thread_idx >= o) {
                shared_mem[thread_idx] += shared_mem[thread_idx - o];
            }
        }
        __syncthreads();
    }
    
    if (thread_idx == 0) {
        shared_mem[0] = 0;
    }
    __syncthreads();
    
    // Scan down
    for (int d = 1; d < dim_size; d <<= 1) {
        int o = d;
        d <<= 1;
        if (thread_idx < d) {
            if (thread_idx >= o) {
                shared_mem[thread_idx] += shared_mem[thread_idx - o];
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (thread_idx < dim_size) {
        output[global_idx] = shared_mem[thread_idx];
    }
}

torch::Tensor scan_cumsum_cuda(torch::Tensor input, int dim) {
    int dim_size = input.size(dim);
    int batch_size = input.size(0);
    int other_dims = 1;
    for (int i = 1; i < input.dim(); i++) {
        if (i != dim) {
            other_dims *= input.size(i);
        }
    }
    
    dim3 threads(dim_size);
    dim3 blocks(other_dims, batch_size, 1);
    
    auto output = torch::empty_like(input);
    
    int smem_size = dim_size * sizeof(float);
    scan_kernel<float><<<blocks, threads, smem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim_size,
        batch_size,
        other_dims
    );
    
    return output;
}

"""

scan_header = "torch::Tensor scan_cumsum_cuda(torch::Tensor input, int dim);"

scan_ops = load_inline(
    name="scan_ops",
    cpp_sources=[scan_header],
    cuda_sources=[scan_kernel],
    functions=["scan_cumsum_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scan_cumsum = scan_ops

    def forward(self, x):
        return self.scan_cumsum.scan_cumsum_cuda(x, self.dim)