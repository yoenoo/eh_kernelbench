import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmin_kernel = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

template <typename scalar_t>
__global__ void argmin_kernel(const scalar_t* __restrict__ data,
                             long* output,
                             int batch_size,
                             int dim_size,
                             int other_dims_size,
                             int dim) {
    // Each block handles a specific (batch, other_dim) index
    int batch_idx = blockIdx.x;
    int other_dim_idx = blockIdx.y;

    // Compute linear index
    int output_idx = batch_idx * other_dims_size + other_dim_idx;

    scalar_t min_val = INFINITY;
    int min_index = -1;

    // Iterate over the dimension
    for (int d = threadIdx.x; d < dim_size; d += blockDim.x) {
        int data_idx = batch_idx * dim_size * other_dims_size +
                       d * other_dims_size + other_dim_idx;
        scalar_t val = data[data_idx];
        if (val < min_val) {
            min_val = val;
            min_index = d;
        }
    }

    // Use reduction in the block to find the min
    __shared__ scalar_t shared_min[32];
    __shared__ int shared_index[32];

    shared_min[threadIdx.x] = min_val;
    shared_index[threadIdx.x] = min_index;

    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s) {
            if (shared_min[threadIdx.x + s] < shared_min[threadIdx.x]) {
                shared_min[threadIdx.x] = shared_min[threadIdx.x + s];
                shared_index[threadIdx.x] = shared_index[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[output_idx] = shared_index[0];
    }
}

torch::Tensor argmin_cuda(torch::Tensor data, int dim) {
    const auto dims = data.sizes().vec();
    int batch_size = dims[0];
    int dim_size = dims[dim];
    int other_dims_size = 1;
    for (size_t i = 1; i < dims.size(); ++i) {
        if (i != dim) {
            other_dims_size *= dims[i];
        }
    }
    
    auto output = torch::empty({batch_size, other_dims_size}, data.options().dtype(torch::kLong));

    int block_size = 256;
    dim3 blocks(batch_size, other_dims_size);
    dim3 threads(block_size);

    AT_DISPATCH_ALL_TYPES(data.scalar_type(), "argmin_cuda", ([&] {
        argmin_kernel<scalar_t><<<blocks, threads>>>(
            data.data<scalar_t>(),
            output.data_ptr<long>(),
            batch_size,
            dim_size,
            other_dims_size,
            dim);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = """
torch::Tensor argmin_cuda(torch::Tensor data, int dim);
"""

argmin_cuda = load_inline(name="argmin_cuda",
                         cpp_sources=cpp_source,
                         cuda_sources=argmin_kernel,
                         functions=["argmin_cuda"],
                         verbose=True)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.argmin_cuda_func = argmin_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return self.argmin_cuda_func.argmin_cuda(x, self.dim)
        else:
            return torch.argmin(x, dim=self.dim)