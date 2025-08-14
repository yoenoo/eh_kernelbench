import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T>
__global__ void matmul_kernel(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
                             int m, int k, int n, int lda, int ldb, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        T sum = 0;
        for (int e = 0; e < k; ++e) {
            sum += a[row * lda + e] * b[e * ldb + col];
        }
        c[row * ldc + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    const int m = a.size(0);
    const int k = a.size(1);
    const int n = b.size(1);

    auto c = torch::empty({m, n}, a.options());

    const int block_size_x = 32;
    const int block_size_y = 8;
    dim3 block(block_size_x, block_size_y);
    dim3 grid((n + block_size_x - 1)/block_size_x, (m + block_size_y - 1)/block_size_y);

    int lda = a.stride(0);
    int ldb = b.stride(0);
    int ldc = c.stride(0);

    matmul_kernel<float><<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        m, k, n, lda, ldb, ldc
    );

    cudaDeviceSynchronize();
    return c;
}
"""

matmul_cuda_header = "torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);"

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_header,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # The original code uses A.T * B.T, which is equivalent to (B * A)^T
        # However, transposing in PyTorch can incur memory overhead. 
        # To optimize further, we directly compute B^T * A^T transposed
        # The result is (B^T * A^T)^T = A * B, so we transpose the output
        result = self.matmul.matmul_cuda(B.t(), A.t())
        return result.t()