import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void min_reduction_kernel(const scalar_t* input, scalar_t* output, int batch_size, int dim1, int dim2, int reduce_dim) {
    int batch_idx = blockIdx.x;
    int remaining_dim = 1;
    if (reduce_dim == 0) {
        remaining_dim = dim1;
    } else if (reduce_dim == 1) {
        remaining_dim = dim2;
    } else {
        // Handle other dimensions if needed
    }
    int linear_index = batch_idx * remaining_dim + threadIdx.x;
    scalar_t min_val = std::numeric_limits<scalar_t>::max();
    for (int i = threadIdx.x; i < (reduce_dim == 0 ? dim1 : dim2); i += blockDim.x) {
        int input_index;
        if (reduce_dim == 0) {
            input_index = batch_idx * dim1 * dim2 + i * dim2 + threadIdx.x;
        } else {
            input_index = batch_idx * dim1 * dim2 + threadIdx.x * dim1 + i;
        }
        scalar_t val = input[input_index];
        if (val < min_val) {
            min_val = val;
        }
    }
    output[linear_index] = min_val;
}

torch::Tensor min_reduction_cuda(torch::Tensor input, int dim) {
    int64_t batch_size = input.size(0);
    int64_t dim1 = input.size(1);
    int64_t dim2 = input.size(2);
    int reduce_dim = dim;

    int output_size;
    if (reduce_dim == 0) {
        output_size = dim1 * dim2;
    } else if (reduce_dim == 1) {
        output_size = batch_size * dim2;
    } else {
        // Handle errors for invalid dims
        output_size = 1;
    }

    auto output = torch::empty({batch_size, (reduce_dim == 0 ? dim2 : dim1)}, input.options());

    const int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);

    min_reduction_kernel<float><<<grid, block>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        dim1, 
        dim2, 
        reduce_dim
    );

    return output;
}
"""

min_reduction_cpp_source = (
    "torch::Tensor min_reduction_cuda(torch::Tensor input, int dim);"
)

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
        super().__init__()
        self.dim = dim
        self.min_reduction = min_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_reduction.min_reduction_cuda(x, self.dim)

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]