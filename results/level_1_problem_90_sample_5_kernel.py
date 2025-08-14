import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cumprod_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

template <typename scalar_t>
__global__ void cumprod_cuda_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim_size,
    const int outer_stride,
    const int inner_stride,
    const int dim
) {
    // Each block handles a specific position along the non-cumprod dimension(s)
    int batch_idx = blockIdx.x / dim_size;
    int pos = blockIdx.x % dim_size;
    
    // Thread index along the cumulative dimension
    int tid = threadIdx.x;

    __shared__ scalar_t shared_data[512]; // Adjust shared memory based on block size

    // Each thread loads a value from input (along the dim)
    // For position 'pos', threads read input[b * outer_stride + pos * inner_stride + ...] ?
    // Need to ensure the indexing is correct based on input shape.

    // For a 2D tensor (batch_size, dim_size), dim=1, the computation is along the columns.
    // The indexing would be input[batch_idx * dim_size + pos]

    // Compute the current element's value and accumulate
    // Assume blockDim.x = dim_size, so each block has a block size equal to dim_size
    // For a given batch element, compute the cumulative product along dim=1

    // Initialize shared memory with input values
    if (tid < dim_size) {
        int global_idx = batch_idx * dim_size + tid;
        shared_data[tid] = input[global_idx];
    } else {
        shared_data[tid] = 1; // Padding not necessary? Adjust based on blockDim.x
    }
    __syncthreads();

    // Perform parallel reduction to compute cumulative product
    // Using a binary traversal approach (bitwise scan)
    for (int s=1; s <= pos; s <<= 1) {
        if (tid >= s) {
            shared_data[tid] += shared_data[tid - s];
        }
        __syncthreads();
    }
    
    // Write back the result for this thread's position in this batch
    output[batch_idx * dim_size + pos] = shared_data[pos];
}

// Define a helper function to select the correct kernel based on data type
at::Tensor cumprod_cuda(at::Tensor input, int dim) {
    const int batch_size = input.size(0);
    const int dim_size = input.size(1); // assuming dim is 1 as per problem setup
    const int outer_stride = dim_size; // rows in case of 2D
    const int inner_stride = 1; // columns
    const int total_elements = batch_size * dim_size;

    auto output = at::empty_like(input);

    dim3 blocks(batch_size, 1, 1); // Each batch in x-dim, dim_size blocks? Wait need to re-examine.

    // The block size should be equal to dim_size to handle all positions in a row
    dim3 threads(dim_size, 1, 1);
    // Assert that dim_size <= 512 (max threads per block), else adjust shared memory size

    AT_DISPATCH_FLOATING_TYPES(input.type(), "cumprod_cuda", ([&] {
        cumprod_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            dim_size,
            outer_stride,
            inner_stride,
            dim);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

cumprod_cpp_source = """
at::Tensor cumprod_cuda(at::Tensor input, int dim);
"""

cumprod_cuda = load_inline(
    name="cumprod_cuda",
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_source,
    functions=["cumprod_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumprod_cuda = cumprod_cuda

    def forward(self, x):
        return self.cumprod_cuda.cumprod_cuda(x, self.dim)