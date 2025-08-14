import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for min reduction along a dimension
min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

template <typename scalar_t>
__global__ void min_reduction_kernel(const scalar_t* __restrict__ input, scalar_t* output, 
                                    int dim_size, int outer_dim, int inner_dim, int dim) {
    extern __shared__ scalar_t shared_data[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int outer = idx / inner_dim;
    int inner = idx % inner_dim;

    if (idx >= outer_dim * inner_dim) return;

    // Load data into shared memory
    scalar_t val = input[outer * dim_size * inner_dim + inner * dim_size + dim];
    shared_data[threadIdx.x] = val;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] = min(shared_data[threadIdx.x], shared_data[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[outer * inner_dim + inner] = shared_data[0];
    }
}

torch::Tensor min_reduction_cuda(torch::Tensor input, int dim) {
    int dims[] = input.sizes().data();
    int ndim = input.dim();
    int reduction_dim = dim;

    // Compute the size of the dimension to reduce
    int dim_size = dims[reduction_dim];

    // Compute outer and inner dimensions
    int outer_dim = 1;
    for (int i = 0; i < reduction_dim; ++i) {
        outer_dim *= dims[i];
    }
    int inner_dim = 1;
    for (int i = reduction_dim + 1; i < ndim; ++i) {
        inner_dim *= dims[i];
    }

    int total_threads = outer_dim * inner_dim;
    int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    // Allocate output tensor
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + reduction_dim);
    auto output = torch::empty(output_sizes, input.options());

    // Calculate shared memory size
    size_t shared_mem_size = block_size * sizeof(float);

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduction_cuda", ([&]{
        min_reduction_kernel<scalar_t><<<num_blocks, block_size, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            outer_dim,
            inner_dim,
            dim);
    }));

    return output;
}
"""

min_reduction_cpp_source = """
torch::Tensor min_reduction_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
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