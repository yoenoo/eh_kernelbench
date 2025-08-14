import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-scalar multiplication
mat_scalar_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void mat_scalar_mul_kernel(const scalar_t* __restrict__ A, scalar_t s, scalar_t* __restrict__ C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * s;
    }
}

std::tuple<torch::Tensor> mat_scalar_mul_cuda(torch::Tensor A, float s) {
    auto size = A.numel();
    auto C = torch::empty({A.size(0), A.size(1)}, A.options());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(A.type(), "mat_scalar_mul_cuda", ([&] {
        mat_scalar_mul_kernel<scalar_t><<<num_blocks, block_size>>>(
            A.data<scalar_t>(), s, C.data<scalar_t>(), size
        );
    }));

    return std::make_tuple(C);
}
"""

mat_scalar_mul_cpp_source = (
    "std::tuple<torch::Tensor> mat_scalar_mul_cuda(torch::Tensor A, float s);"
)

# Compile the inline CUDA code for matrix-scalar multiplication
mat_scalar_mul = load_inline(
    name="mat_scalar_mul",
    cpp_sources=mat_scalar_mul_cpp_source,
    cuda_sources=mat_scalar_mul_source,
    functions=["mat_scalar_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.mat_scalar_mul = mat_scalar_mul

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.mat_scalar_mul.mat_scalar_mul_cuda(A, s)[0]

def get_inputs():
    M = 16384 * 4
    N = 4096 * 4
    A = torch.rand(M, N).cuda()
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []  # No special initialization inputs needed