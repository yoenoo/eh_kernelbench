import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for lower triangular matrix multiplication
triangular_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ 
void triangular_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) {
        return;
    }

    float sum = 0.0;
    for (int k = 0; k < N; k++) {
        // Only compute if A[row][k] and B[k][col] are on or below the diagonal
        // For lower triangular matrices: row >= k for A (since A is stored in row-major)
        // and k >= col for B (since B columns are accessed from left to right)
        if (row >= k && k >= col) {
            sum += A[row * N + k] * B[k * N + col];
        }
    }
    if (row >= col) {  // Result is also lower triangular
        C[row * N + col] = sum;
    }
}

torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    const int block_size = 32;
    dim3 threads(block_size, block_size);
    dim3 blocks((N + block_size - 1)/block_size, (N + block_size - 1)/block_size);

    auto C = torch::zeros({N, N}, A.options());

    triangular_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    cudaDeviceSynchronization();
    return C;
}
"""

triangular_matmul_cpp_source = "torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);"

triangular_matmul = load_inline(
    name="triangular_matmul",
    cpp_sources=triangular_matmul_cpp_source,
    cuda_sources=triangular_matmul_source,
    functions=["triangular_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.triangular_matmul = triangular_matmul

    def forward(self, A, B):
        return self.triangular_matmul.triangular_matmul_cuda(A.cuda(), B.cuda())