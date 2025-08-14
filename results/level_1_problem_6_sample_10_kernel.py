import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_large_k_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_DIM 32
#define BLOCK_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0;
        for (int k = 0; k < K; k += TILE_DIM) {
            __shared__ float shared_A[TILE_DIM][TILE_DIM];
            __shared__ float shared_B[TILE_DIM][TILE_DIM];

            int a_row = row;
            int a_col = k + threadIdx.y;
            int b_row = k + threadIdx.x;
            int b_col = col;

            if (a_col < K) {
                shared_A[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
            } else {
                shared_A[threadIdx.y][threadIdx.x] = 0.0;
            }

            if (b_col < N && b_row < K) {
                shared_B[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
            } else {
                shared_B[threadIdx.y][threadIdx.x] = 0.0;
            }

            __syncthreads();

            for (int i = 0; i < TILE_DIM; ++i) {
                sum += shared_A[threadIdx.y][i] * shared_B[i][threadIdx.x];
            }

            __syncthreads();
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_large_k_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor C = torch::zeros({M, N}, options);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1)/BLOCK_SIZE, (M + BLOCK_SIZE - 1)/BLOCK_SIZE);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}
"""

matmul_large_k_cpp_source = "torch::Tensor matmul_large_k_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for large K matrix multiplication
matmul_large_k = load_inline(
    name="matmul_large_k",
    cpp_sources=matmul_large_k_cpp_source,
    cuda_sources=matmul_large_k_source,
    functions=["matmul_large_k_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_large_k = matmul_large_k

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_large_k.matmul_large_k_cuda(A, B)