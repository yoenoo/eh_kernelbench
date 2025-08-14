import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

scalar_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void scalar_mult_kernel(const scalar_t* A, scalar_t s, scalar_t* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * s;
    }
}

torch::Tensor scalar_mult_cuda(torch::Tensor A, float s) {
    auto output = torch::empty_like(A);
    int size = A.numel();
    int threads_per_block = 1024;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(A.type(), "scalar_mult_cuda", ([&] {
        scalar_mult_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            A.data<scalar_t>(), s, output.data<scalar_t>(), size);
    }));

    return output;
}
"""

scalar_mult_cpp_source = """
torch::Tensor scalar_mult_cuda(torch::Tensor A, float s);
"""

scalar_mult = load_inline(
    name="scalar_mult",
    cpp_sources=scalar_mult_cpp_source,
    cuda_sources=scalar_mult_source,
    functions=["scalar_mult_cuda"],
    verbose=True,
    extra_cflags=["-DVERSION_GE_1_5"],
    extra_cuda_cflags=["-std=c++14", "--expt-extended-lambda"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.scalar_mult = scalar_mult

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.scalar_mult.scalar_mult_cuda(A.cuda(), s).to(A.device)