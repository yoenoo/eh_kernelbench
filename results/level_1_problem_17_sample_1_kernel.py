import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

__global__ void matrix_mult_kernel(const float* A, const float* B, float* C, const int M, const int K, const int N) {
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

torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B, int M, int K, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    auto C = torch::empty({M, N}, A.options());

    matrix_mult_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    cudaDeviceSynchronize();

    return C;
}
"""

matrix_mult_cpp_source = "torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B, int M, int K, int N);"

matrix_mult = load_inline(
    name="matrix_mult",
    cpp_sources=matrix_mult_cpp_source,
    cuda_sources=matrix_mult_source,
    functions=["matrix_mult_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix_mult = matrix_mult

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.size()
        _, N = B.size()
        return self.matrix_mult.matrix_mult_cuda(A, B, M, K, N)