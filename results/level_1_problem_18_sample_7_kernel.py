import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class MatrixMulCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        M, K = A.size()
        K2, N = B.size()
        assert K == K2, "Matrix dimensions must be compatible for multiplication"
        C = torch.empty((M, N), device='cuda', dtype=A.dtype)
        
        # Define block size
        TILE_DIM = 32
        BLOCK_SIZE = 32

        # CUDA kernel parameters
        threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
        blocks_per_grid = (
            (M + threads_per_block[0] - 1) // threads_per_block[0],
            (N + threads_per_block[1] - 1) // threads_per_block[1]
        )

        matmul_kernel[blocks_per_grid, threads_per_block](A, B, C, M, K, N)
        
        return C

# CUDA kernel implementation using tiled matrix multiplication
matmul_kernel_source = """
#include <iostream>
#include <torch/extension.h>

__constant__ float A_shared[32][32];
__constant__ float B_shared[32][32];

__global__ void matmul_kernel(torch::PackedTensorAccessor32<float,2> A,
                             torch::PackedTensorAccessor32<float,2> B,
                             torch::PackedTensorAccessor32<float,2> C,
                             int M, int K, int N) {
    __shared__ float SharedA[TILE_DIM][TILE_DIM + 1];
    __shared__ float SharedB[TILE_DIM][TILE_DIM + 1];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0;

    for (int tile_k = 0; tile_k < (K - 1)/TILE_DIM + 1; tile_k++) {
        // Load tiles into shared memory
        if (tile_k * TILE_DIM + threadIdx.x < K && row < M) {
            SharedA[threadIdx.y][threadIdx.x] = A[row][tile_k * TILE_DIM + threadIdx.x];
        } else {
            SharedA[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (tile_k * TILE_DIM + threadIdx.y < K && col < N) {
            SharedB[threadIdx.x][threadIdx.y] = B[tile_k * TILE_DIM + threadIdx.y][col];
        } else {
            SharedB[threadIdx.x][threadIdx.y] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            sum += SharedA[threadIdx.y][k] * SharedB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row][col] = sum;
    }
}

#define TILE_DIM 32

"""
matmul_kernel = load_inline(
    name="matmul_kernel",
    cpp_sources="",
    cuda_sources=matmul_kernel_source,
    functions=[],
    verbose=False,
    with_cuda=True
).matmul_kernel

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return MatrixMulCUDA.apply(A.cuda(), B.cuda())

# Maintain the same input generation functions
def get_inputs():
    A = torch.rand(K, M).cuda()
    B = torch.rand(N, K).cuda()
    return [A.t(), B.t()]  # Transpose to match PyTorch's matmul order

def get_init_inputs():
    return []