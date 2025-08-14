import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void custom_matmul_kernel(const float* A, const float* B, float* C, 
                                    int M, int K, int N) {
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0;

    for (int block_idx = 0; block_idx < (K + TILE_DIM - 1)/TILE_DIM; block_idx++) {
        __shared__ float shared_A[TILE_DIM][TILE_DIM + 1];
        __shared__ float shared_B[TILE_DIM][TILE_DIM + 1];

        int a_col = block_idx * TILE_DIM + threadIdx.x;
        int b_row = block_idx * TILE_DIM + threadIdx.y;

        if (a_col < K && row < M) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (b_row < K && col < N) {
            shared_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_DIM; k++) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor custom_matmul(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor C = torch::zeros({M, N}, options);

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1)/TILE_DIM, (M + TILE_DIM - 1)/TILE_DIM);

    custom_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                             C.data_ptr<float>(), M, K, N);
    cudaDeviceSynchronize();
    return C;
}
"""

matmul_cpp_src = "torch::Tensor custom_matmul(torch::Tensor A, torch::Tensor B);"

custom_matmul = load_inline(
    name='custom_matmul',
    cpp_sources=matmul_cpp_src,
    cuda_sources=matmul_source,
    functions=['custom_matmul'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = custom_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.custom_matmul(A, B)