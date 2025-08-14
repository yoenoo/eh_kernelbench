cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* __restrict__ a,
                             const scalar_t* __restrict__ b,
                             scalar_t* __restrict__ c,
                             const int m,
                             const int k,
                             const int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        scalar_t sum = 0;
        for (int e = 0; e < k; ++e) {
            sum += a[row * k + e] * b[e * n + col];
        }
        c[row * n + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    const int m = a.size(0);
    const int k = a.size(1);
    const int n = b.size(1);

    auto c = torch::empty({m, n}, a.options());

    const int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((n + block_size - 1) / block_size,
             (m + block_size - 1) / block_size);

    matmul_kernel<float><<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, k, n);

    return c;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);"

# Compile the inline CUDA code for matrix multiplication
matmul_ext = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul_ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A, B)

def get_inputs():
    # Use the same input dimensions as defined globally
    M = 16384 * 2
    N = 16384 * 2
    K = 32 * 2
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []