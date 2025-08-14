import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <vector>
#include <math.h>

// CUDA kernel for matrix-vector multiplication
__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* C, const int M, const int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < 1) {
        float sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k];
        }
        C[row * 1 + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int threads_per_block = 32;
    dim3 threads(threads_per_block, threads_per_block);

    // Compute grid dimensions
    dim3 blocks((1 + threads_per_block - 1) / threads_per_block, (M + threads_per_block - 1) / threads_per_block, 1);

    torch::Tensor C = torch::zeros({M, 1}, A.options());

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K);

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

matmul_ext = load_inline(
    name="matmul_ext",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A.cuda(), B.cuda())