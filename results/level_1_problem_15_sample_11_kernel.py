import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for lower triangular matrix multiplication
tril_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tril_matmul_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= M) return;
    
    if (row < col) {
        c[row * M + col] = 0.0;
        return;
    }
    
    scalar_t sum = 0.0;
    for (int k = 0; k < M; ++k) {
        sum += a[row * M + k] * b[k * M + col];
    }
    c[row * M + col] = sum;
}

torch::Tensor tril_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    int M = a.size(0);
    const int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((M + block_size - 1)/block_size, (M + block_size - 1)/block_size);

    auto c = torch::zeros({M, M}, a.options());
    
    AT_DISPATCH_ALL_TYPES(a.scalar_type(), "tril_matmul_cuda", ([&] {
        tril_matmul_kernel<scalar_t><<<grid, block>>>(
            a.data<scalar_t>(),
            b.data<scalar_t>(),
            c.data<scalar_t>(),
            M);
    }));
    
    return c;
}
"""

tril_matmul_cpp_source = "torch::Tensor tril_matmul_cuda(torch::Tensor a, torch::Tensor b);"

# Compile the custom CUDA kernel
tril_matmul = load_inline(
    name="tril_matmul",
    cpp_sources=tril_matmul_cpp_source,
    cuda_sources=tril_matmul_source,
    functions=["tril_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tril_matmul = tril_matmul

    def forward(self, A, B):
        return self.tril_matmul.tril_matmul_cuda(A, B)

def get_inputs():
    M = 4096
    A = torch.rand(M, M).cuda()
    B = torch.rand(M, M).cuda()
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]

def get_init_inputs():
    return []