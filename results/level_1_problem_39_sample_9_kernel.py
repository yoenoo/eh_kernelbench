import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for L2 normalization
l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void l2_norm_kernel(const scalar_t* x, scalar_t* out, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int elem_idx = threadIdx.x;

    __shared__ scalar_t shared_data[1024]; // Adjust based on maximum dimension

    if (elem_idx < dim) {
        scalar_t val = x[batch_idx * dim + elem_idx];
        shared_data[elem_idx] = val * val;
    }
    __syncthreads();

    // Compute the sum of squares for each batch
    scalar_t sum_squares = 0.0;
    for (int i = 0; i < dim; ++i) {
        if (threadIdx.x == 0 && i == elem_idx) {
            sum_squares += shared_data[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        scalar_t norm = sqrt(sum_squares);
        for (int i = 0; i < dim; ++i) {
            out[batch_idx * dim + i] = x[batch_idx * dim + i] / norm;
        }
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int dim = x.size(1);

    auto out = torch::empty_like(x);

    dim3 blocks(batch_size);
    dim3 threads(dim < 1024 ? dim : 1024); // Limit threads to 1024

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "l2_norm_cuda", ([&] {
        l2_norm_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            dim);
    }));

    return out;
}
"""

l2_norm_cpp_source = (
    "torch::Tensor l2_norm_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code
l2_norm = load_inline(
    name="l2_norm",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_norm = l2_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2_norm.l2_norm_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []