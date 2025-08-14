import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void mean_reduction_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out, int dim_size, int other_size, int dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= other_size) return;

    scalar_t sum = 0;
    for (int d = 0; d < dim_size; ++d) {
        int offset = index * dim_size + d;
        sum += x[offset];
    }
    out[index] = sum / dim_size;
}

torch::Tensor mean_reduction_cuda(torch::Tensor x, int dim) {
    int dim_size = x.size(dim);
    int other_size = x.numel() / dim_size;

    auto out = torch::empty({other_size}, x.options());
    const int block_size = 256;
    const int num_blocks = (other_size + block_size - 1) / block_size;

    // Select the appropriate kernel based on data type
    AT_DISPATCH_FLOATING_TYPES(x.type(), "mean_reduction_cuda", ([&] {
        mean_reduction_kernel<scalar_t><<<num_blocks, block_size>>>(
            x.data_ptr<scalar_t>(), 
            out.data_ptr<scalar_t>(), 
            dim_size,
            other_size,
            dim);
    }));

    return out.reshape(x.sizes().remove_dim(dim));
}
"""

mean_reduction_cpp_source = "torch::Tensor mean_reduction_cuda(torch::Tensor x, int dim);"

mean_reduction = load_inline(
    name="mean_reduction",
    cpp_sources=mean_reduction_cpp_source,
    cuda_sources=mean_reduction_source,
    functions=["mean_reduction_cuda"],
    verbose=True,
    extra_cflags=["-g"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mean_reduction = mean_reduction

    def forward(self, x):
        return self.mean_reduction.mean_reduction_cuda(x, self.dim)

def get_inputs():
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]

def get_init_inputs():
    return []