import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void optimized_matmul(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
                                int M, int N, int K) {
    // Block and thread indices
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Each block computes a tile of the result
    __shared__ T shared_A[TILE_DIM][TILE_DIM];
    __shared__ T shared_B[TILE_DIM][TILE_DIM];

    int Row = by * TILE_DIM + ty;
    int Col = bx * TILE_DIM + tx;
    T sum = 0;

    for (int m = 0; m < (K - 1)/TILE_DIM + 1; ++m) {
        // Load tiles from A and B into shared memory
        int a_row = Row;
        int a_col = m * TILE_DIM + tx;
        shared_A[ty][tx] = (a_col < K) ? A[a_row * K + a_col] : 0;

        int b_row = m * TILE_DIM + ty;
        int b_col = Col;
        shared_B[ty][tx] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0;

        __syncthreads();

        // Compute the dot product for this tile
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }
        __syncthreads();
    }

    if (Row < M && Col < N) {
        C[Row * N + Col] = sum;
    }
}

#define TILE_DIM 32

torch::Tensor optimized_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1)/TILE_DIM, (M + TILE_DIM - 1)/TILE_DIM);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "optimized_matmul_cuda", ([&]{
        optimized_matmul<scalar_t><<<blocks, threads>>>(A.data<scalar_t>(), B.data<scalar_t>(), C.data<scalar_t>(), M, N, K);
    }));

    return C;
}
"""

cpp_source = """
torch::Tensor optimized_matmul_cuda(torch::Tensor, torch::Tensor);
"""

# Compile the CUDA kernel
optimized_matmul = load_inline(
    name="optimized_matmul",
    cpp_sources=cpp_source,
    cuda_sources=matmul_kernel_source,
    functions=["optimized_matmul_cuda"],
    verbose=True,
    extra_cflags=["-DHIP_ENABLE_HIP=0"],
    extra_cuda_cflags=["-arch=sm_80"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.optimized_matmul = optimized_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are transposed correctly
        return self.optimized_matmul.optimized_matmul_cuda(A.t(), B.t())