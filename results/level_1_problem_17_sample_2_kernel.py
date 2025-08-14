import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication (A * B^T)
matmul_transposed_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_transposed_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transposed_cuda(
    torch::Tensor A,
    torch::Tensor B,
    int M,
    int N,
    int K
) {
    const int TPB = 32;

    dim3 threads(TPB, TPB);
    dim3 blocks(
        (N + TPB - 1) / TPB,
        (M + TPB - 1) / TPB
    );

    auto C = torch::empty({M, N}, torch::CUDA(A.device().index()));

    matmul_transposed_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        N,
        K
    );

    return C;
}
"""

matmul_transposed_cpp_source = (
    "torch::Tensor matmul_transposed_cuda(torch::Tensor A, torch::Tensor B, int M, int N, int K);"
)

# Compile the inline CUDA code
matmul_transposed = load_inline(
    name="matmul_transposed",
    cpp_sources=matmul_transposed_cpp_source,
    cuda_sources=matmul_transposed_source,
    functions=["matmul_transposed_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_transposed = matmul_transposed

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M = A.size(0)
        N = B.size(0)
        K = A.size(1)
        return self.matmul_transposed.matmul_transposed_cuda(A.cuda(), B.cuda(), M, N, K)