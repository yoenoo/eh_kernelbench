import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrixmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matrixmul_cuda(torch::Tensor A, torch::Tensor B, int N) {
    auto threads_per_block = dim3(16, 16);
    auto blocks_per_grid = dim3(
        (N + threads_per_block.x - 1) / threads_per_block.x,
        (N + threads_per_block.y - 1) / threads_per_block.y
    );

    auto C = torch::empty(N, N, torch::device("cuda"));

    matrixmul_kernel<<<blocks_per_grid, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matrixmul_cuda(torch::Tensor A, torch::Tensor B, int N);"

# Compile the custom kernel
matrixmul = load_inline(
    name="matrixmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matrixmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrixmul = matrixmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        N = A.size(0)
        return self.matrixmul.matrixmul_cuda(A.cuda(), B.cuda(), N)