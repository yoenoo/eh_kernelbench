import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for masked cumsum
masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void masked_cumsum_kernel(const scalar_t* x, const unsigned char* mask, scalar_t* out, int dim_size, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int elem_idx = threadIdx.x;

    if (elem_idx >= dim_size) return;

    int offset = batch_idx * dim_size;
    scalar_t sum = 0;
    for (int i = 0; i <= elem_idx; ++i) {
        int pos = offset + i;
        if (mask[pos]) {
            sum += x[pos];
        }
    }
    out[offset + elem_idx] = sum;
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim) {
    auto dim_size = x.size(dim);
    auto batch_size = x.size(0);
    auto out = torch::empty_like(x);

    const int block_size = 1024;  // Threads per block (assuming dim_size <= 1024 for simplicity)
    const int grid_size = batch_size;  // One block per batch element

    auto x_data = x.data<scalar_t>();
    auto mask_data = mask.data<uint8_t>();
    auto out_data = out.data<scalar_t>();

    masked_cumsum_kernel<scalar_t><<<grid_size, block_size>>>(
        x_data, mask_data, out_data, dim_size, batch_size, dim);

    return out;
}
"""

masked_cumsum_cpp_source = R""" 
torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim);
"""

# Compile the CUDA kernel
masked_cumsum = load_inline(
    name="masked_cumsum",
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        return masked_cumsum.masked_cumsum_cuda(x, mask, self.dim)