import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for cumulative product
cumprod_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void cumprod_kernel(scalar_t* __restrict__ out, const scalar_t* __restrict__ in, int64_t size, int64_t dim_size, int64_t outer_dim, int64_t inner_dim) {
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    
    int64_t outer = index / (dim_size * inner_dim);
    int64_t dim_index = (index / inner_dim) % dim_size;
    int64_t inner = index % inner_dim;

    int64_t in_idx = outer * dim_size * inner_dim + dim_index * inner_dim + inner;
    out[index] = 1;

    // Compute cumulative product along the dimension
    for (int i = 0; i <= dim_index; ++i) {
        out[index] *= in[outer * dim_size * inner_dim + i * inner_dim + inner];
    }
}

std::tuple<torch::Tensor, torch::Tensor> cumprod_cuda(torch::Tensor in, int64_t dim) {
    auto out = torch::empty_like(in);
    auto in_contig = in.contiguous();
    auto output_contig = out_contiguous(in_contig.sizes());

    int64_t dim_size = in.size(dim);
    int64_t batch_dims = 1;
    for (int64_t i = 0; i < dim; ++i) {
        batch_dims *= in.size(i);
    }
    int64_t inner_dims = 1;
    for (int64_t i = dim + 1; i < in.dim(); ++i) {
        inner_dims *= in.size(i);
    }
    int64_t total_elements = in.numel();

    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    AT_DISPATCH_ALL_TYPES_AND2(in.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_kernel<scalar_t><<<grid_size, block_size>>>(
            out.data_ptr<scalar_t>(),
            in.data_ptr<scalar_t>(),
            total_elements,
            dim_size,
            batch_dims,
            inner_dims);
    }));

    cudaDeviceSynchronize();
    return std::make_tuple(out, in);
}

"""

# Compile the inline CUDA code
cumprod_module = load_inline(
    name='cumprod',
    cpp_sources="",
    cuda_sources=cumprod_source,
    functions=['cumprod_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumprod_cuda = cumprod_module

    def forward(self, x):
        return self.cumprod_cuda.cumprod_cuda(x, self.dim)[0]