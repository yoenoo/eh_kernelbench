import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matrix_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void matrix_mult_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    __shared__ float shared_A[TILE_DIM][TILE_DIM + 1];
    __shared__ float shared_B[TILE_DIM][TILE_DIM + 1];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float C_value = 0.0f;

    for (int tile_k = 0; tile_k < (K + TILE_DIM - 1) / TILE_DIM; ++tile_k) {
        // Load tiles of A and B into shared memory
        int a_row = row;
        int a_col = tile_k * TILE_DIM + threadIdx.x;
        bool a_in_bounds = (a_col < K);
        shared_A[threadIdx.y][threadIdx.x] = 
            a_in_bounds ? A[a_row * K + a_col] : 0.0f;

        int b_row = tile_k * TILE_DIM + threadIdx.y;
        int b_col = col;
        bool b_in_bounds = (b_row < K);
        shared_B[threadIdx.y][threadIdx.x] = 
            b_in_bounds ? B[b_row * N + b_col] : 0.0f;

        __syncthreads();

        // Compute the dot product
        for (int k = 0; k < TILE_DIM; ++k) {
            if (a_in_bounds && b_in_bounds) {
                C_value += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = C_value;
    }
}

torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y * BLOCK_ROWS - 1) / (threads.y * BLOCK_ROWS)
    );

    matrix_mult_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}
"""

matrix_mul_cpp_source = "torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for matrix multiplication
matrix_mult = load_inline(
    name="matrix_mult",
    cpp_sources=matrix_mul_cpp_source,
    cuda_sources=matrix_mul_source,
    functions=["matrix_mult_cuda"],
    verbose=False,
    extra_cflags=["-g", "-w"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_mult = matrix_mult

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrix_mult.matrix_mult_cuda(A.cuda(), B.cuda())