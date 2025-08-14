import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void min_reduction_kernel(const scalar_t* input, scalar_t* output, int dim_size, int outer_size, int inner_size, int dim) {
    extern __shared__ scalar_t shared_buf[];
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    
    // Each block processes a slice along the reduction dimension
    scalar_t min_val = std::numeric_limits<scalar_t>::max();
    
    // Compute the stride for each element in the reduction dimension
    for (int i = block_idx * blockDim.x + thread_idx; i < dim_size; i += gridDim.x * blockDim.x) {
        int index = i * outer_size * inner_size;
        for (int j = 0; j < outer_size; j++) {
            for (int k = 0; k < inner_size; k++) {
                int pos = index + j * inner_size + k;
                if (input[pos] < min_val) {
                    min_val = input[pos];
                }
            }
            index += dim_size * inner_size;  // Move to next outer element
        }
    }
    
    // Use shared memory for block-wise reduction
    shared_buf[thread_idx] = min_val;
    __syncthreads();
    
    // Perform block-wide reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_idx < s) {
            if (shared_buf[thread_idx + s] < shared_buf[thread_idx]) {
                shared_buf[thread_idx] = shared_buf[thread_idx + s];
            }
        }
        __syncthreads();
    }
    
    if (thread_idx == 0) {
        output[block_idx] = shared_buf[0];
    }
}

torch::Tensor min_reduction_cuda(torch::Tensor input, int dim) {
    int dims[] = {input.size(0), input.size(1), input.size(2)};
    int dim_size = input.size(dim);
    int outer_size = 1, inner_size = 1;
    if (dim == 0) {
        outer_size = 1;
        inner_size = dims[1] * dims[2];
    } else if (dim == 1) {
        outer_size = dims[0];
        inner_size = dims[2];
    } else {
        outer_size = dims[0] * dims[1];
        inner_size = 1;
    }
    
    int output_size = 1;
    for (int i = 0; i < input.dim(); i++) {
        if (i != dim) output_size *= input.size(i);
    }
    
    auto output = torch::empty({output_size}, input.options());
    
    int block_size = 256;
    int grid_size = std::min(32768, (dim_size + block_size - 1) / block_size);
    
    auto stream = at::cuda::getCurrentCUDAStream();
    
    dim3 blocks(grid_size);
    dim3 threads(block_size);
    size_t shared_mem = threads.x * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "min_reduction_cuda", ([&] {
        min_reduction_kernel<scalar_t><<<blocks, threads, shared_mem, stream>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            outer_size,
            inner_size,
            dim
        );
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

min_reduction_cpp_source = """
torch::Tensor min_reduction_cuda(torch::Tensor input, int dim);
"""

min_reduction = load_inline(
    name="min_reduction",
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_source,
    functions=["min_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.min_reduction = min_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_reduction.min_reduction_cuda(x, self.dim)