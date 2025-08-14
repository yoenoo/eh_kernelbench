import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void fast_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* C, int M, int K, int N) {
    __shared__ float shared_A[TILE_DIM][TILE_DIM + 1];
    __shared__ float shared_B[TILE_DIM][TILE_DIM + 1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_DIM * BLOCK_ROWS + ty * TILE_DIM + tx;
    int Col = bx * TILE_DIM * BLOCK_ROWS + tx * TILE_DIM + ty;

    float val = 0.0;
    for (int p = 0; p < (K - 1) / (TILE_DIM * BLOCK_ROWS) + 1; p++) {
        // Load tiles of A and B into shared memory
        int a_row = by * TILE_DIM * BLOCK_ROWS + ty * TILE_DIM + tx;
        int a_col = p * TILE_DIM * BLOCK_ROWS + tx;
        shared_A[ty][tx] = (a_col < K) ? A[a_row * K + a_col] : 0.0f;
        
        int b_row = p * TILE_DIM * BLOCK_ROWS + ty;
        int b_col = bx * TILE_DIM * BLOCK_ROWS + tx * TILE_DIM + ty;
        shared_B[ty][tx] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; k++) {
            val += shared_A[ty][k] * shared_B[k][tx];
        }
        __syncthreads();
    }

    if (Row < M && Col < N) {
        C[Row * N + Col] = val;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor C = torch::zeros({M, N}, options);

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1)/TILE_DIM, (M + TILE_DIM - 1)/TILE_DIM);

    fast_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}
"""

matmul_kernel_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=[matmul_kernel_cpp_source],
    cuda_sources=[matmul_kernel_source],
    functions=["matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A, B)