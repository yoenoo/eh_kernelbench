import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void min_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int dim1,
    int dim2,
    int reduce_dim
) {
    int batch_idx = blockIdx.x;
    int remaining_dim = reduce_dim == 0 ? dim2 : (reduce_dim == 1 ? dim2 : dim1);
    int output_idx = batch_idx * remaining_dim + threadIdx.x;

    __shared__ scalar_t shared_mem[1024];
    int tid = threadIdx.x;
    scalar_t min_val = std::numeric_limits<scalar_t>::max();

    for (int i = tid; i < (reduce_dim == 0 ? 1 : (reduce_dim == 1 ? dim1 : dim2)); i += blockDim.x) {
        int input_idx;
        if (reduce_dim == 0) {
            input_idx = 0 * dim1 * dim2 + threadIdx.x * dim2 + i; // Not sure about the exact indexing here
        } 
        else if (reduce_dim == 1) {
            input_idx = batch_idx * dim1 * dim2 + i * dim2 + threadIdx.x;
        } 
        else {
            input_idx = batch_idx * dim1 * dim2 + threadIdx.x * dim1 + i;
        }
        min_val = fminf(min_val, input[input_idx]);
    }

    shared_mem[tid] = min_val;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            shared_mem[tid] = fminf(shared_mem[tid], shared_mem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[output_idx] = shared_mem[0];
    }
}

std::tuple<torch::Tensor> min_reduction_cuda(torch::Tensor input, int dim) {
    const int batch_size = input.size(0);
    const int dim1 = input.size(1);
    const int dim2 = input.size(2);
    int reduce_dim = dim;

    int output_size[3];
    if (reduce_dim == 0) {
        output_size[0] = 1;
        output_size[1] = dim1;
        output_size[2] = dim2;
    } else if (reduce_dim == 1) {
        output_size[0] = batch_size;
        output_size[1] = 1;
        output_size[2] = dim2;
    } else {
        output_size[0] = batch_size;
        output_size[1] = dim1;
        output_size[2] = 1;
    }

    auto output = torch::empty({output_size[0], output_size[1], output_size[2]}, input.options());

    const int block_size = 1024;
    const int grid_size = batch_size;

    min_reduction_kernel<<<grid_size, block_size>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size,
        dim1,
        dim2,
        reduce_dim
    );

    return output;
}
"""

min_reduction_cpp_source = """
std::tuple<torch::Tensor> min_reduction_cuda(torch::Tensor input, int dim);
"""

min_reduction = load_inline(
    name="min_reduction",
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_source,
    functions=["min_reduction_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.min_reduction = min_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_reduction.min_reduction_cuda(x, self.dim)[0]