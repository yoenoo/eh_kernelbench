cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void min_reduction_kernel(const scalar_t* __restrict__ input, scalar_t* output, int batch_size, int dim1, int dim2, int reduce_dim) {
    int batch_idx = blockIdx.x;
    int out_index = threadIdx.x;
    int in_index = batch_idx * dim1 * dim2 + out_index * dim2;
    
    // Compute the offset for the reduction dimension
    int stride = (reduce_dim == 0) ? 1 : (reduce_dim == 1) ? dim2 : dim1;
    int start = in_index;
    int end = start + stride * ((reduce_dim == 0) ? dim1 : (reduce_dim == 1 ? dim2 : dim1));
    
    scalar_t min_val = std::numeric_limits<scalar_t>::max();
    for (int i = start; i < end; i += stride) {
        if (input[i] < min_val) {
            min_val = input[i];
        }
    }
    output[batch_idx * dim2 + out_index] = min_val;
}

torch::Tensor min_reduction_cuda(torch::Tensor input, int dim) {
    int batch_size = input.size(0);
    int dim1 = input.size(1);
    int dim2 = input.size(2);
    auto output = torch::empty({batch_size, dim2}, input.options());
    
    dim = dim < 0 ? dim + input.dim() : dim;
    
    int blocks = batch_size;
    int threads = dim1;
    
    // Launch kernel based on reduction dimension
    if (dim == 0) {
        min_reduction_kernel<float><<<blocks, threads>>>(
            input.data_ptr<float>(), 
            output.data_ptr<float>(), 
            batch_size, dim1, dim2, dim
        );
    } else if (dim == 1) {
        min_reduction_kernel<float><<<blocks, threads>>>(
            input.data_ptr<float>(), 
            output.data_ptr<float>(), 
            batch_size, dim1, dim2, dim
        );
    } else {
        throw std::runtime_error("Invalid dimension");
    }
    
    return output;
}
"""

min_reduction_cpp_source = "torch::Tensor min_reduction_cuda(torch::Tensor input, int dim);"

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