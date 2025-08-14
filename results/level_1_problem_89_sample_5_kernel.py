import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import numpy as np

def get_block_size():
    return 256  # Define an optimal block size for the CUDA kernel

# CUDA kernel for inclusive prefix sum (cumsum) along the specified dimension
scan_kernel_code = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector_types.hpp>

template <typename scalar_t>
__global__ void inclusive_scan_kernel(const scalar_t* __restrict__ input,
                                     scalar_t* __restrict__ output,
                                     int64_t total_elements,
                                     int64_t dim_size,
                                     int64_t outer_dim,
                                     int64_t inner_dim) {
    extern __shared__ scalar_t shared_data[];
    int block_start = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int global_tid = block_start + tid;
    
    // Load input into shared memory
    if (global_tid < total_elements) {
        int outer = global_tid / dim_size;
        int local = global_tid % dim_size;
        shared_data[threadIdx.x] = input[global_tid];
    }
    __syncthreads();
    
    // Perform parallel scan within the block
    for (int s = 1; s <= blockDim.x; s *= 2) {
        int ai = 2 * s * tid - s;
        if (ai >= 0 && ai + s < blockDim.x) {
            shared_data[ai + s] += shared_data[ai];
        }
        __syncthreads();
    }
    
    // Write results back
    if (global_tid < total_elements) {
        output[global_tid] = shared_data[threadIdx.x];
    }
}

std::vector<torch::Tensor> inclusive_scan_cuda(torch::Tensor input,
                                              int64_t dim,
                                              int64_t block_size = 256) {
    auto output = torch::empty_like(input);
    
    int64_t dim_size = input.size(dim);
    auto total_elements = input.numel();
    auto num_blocks = (total_elements + block_size - 1) / block_size;
    dim3 blocks(num_blocks);
    dim3 threads(block_size);
    
    int64_t outer_dim = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_dim *= input.size(i);
    }
    int64_t inner_dim = 1;
    for (int64_t i = dim + 1; i < input.dim(); i++) {
        inner_dim *= input.size(i);
    }
    
    auto stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "inclusive_scan_cuda", ([&] {
        inclusive_scan_kernel<scalar_t><<<blocks, threads, block_size * sizeof(scalar_t), stream>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total_elements,
            dim_size,
            outer_dim,
            inner_dim);
    }));
    
    cudaDeviceSynchronize();
    return {output};
}
"""

cpp_source = """
std::vector<torch::Tensor> inclusive_scan_cuda(torch::Tensor input, int64_t dim, int64_t block_size);
"""

# Compile the CUDA code
scan_op = load_inline(
    name="inclusive_scan",
    cpp_sources=cpp_source,
    cuda_sources=scan_kernel_code,
    functions=["inclusive_scan_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.block_size = get_block_size()
    
    def forward(self, x):
        return scan_op.inclusive_scan_cuda(x, self.dim, self.block_size)[0]