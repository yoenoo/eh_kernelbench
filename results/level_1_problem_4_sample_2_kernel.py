import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Matrix dimensions based on the problem statement
M = 256 * 8  # 2048
K = 131072 * 8  # 1048576

# Custom CUDA kernel for matrix-vector multiplication
matrix_vec_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void matvec_mult_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K
) {
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int block_col = 0; block_col < (K + BLOCK_SIZE - 1)/BLOCK_SIZE; ++block_col) {
        int a_row = row;
        int a_col = block_col * BLOCK_SIZE + threadIdx.x;

        int b_row = a_col;
        int b_col = 0; // since B is (K,1)

        bool a_valid = (a_col < K);
        bool b_valid = (b_row < K);

        if (threadIdx.x < K && a_valid)
            shared_A[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        else
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (threadIdx.x < 1 && b_valid)
            shared_B[threadIdx.y] = B[b_row * 1 + b_col];
        else
            shared_B[threadIdx.y] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += shared_A[threadIdx.y][k] * shared_B[k];
        }

        __syncthreads();
    }

    if (row < M && col < 1) {
        C[row * 1 + col] = sum;
    }
}

torch::Tensor matvec_mult_cuda(
    torch::Tensor A,
    torch::Tensor B
) {
    const int M = A.size(0);
    const int K = A.size(1);

    auto C = torch::empty({M, 1}, A.options());

    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE, 16);  // Threads per block
    dim3 grid(1, (M + block.y - 1)/block.y);  // Blocks per grid

    matvec_mult_kernel<BLOCK_SIZE><<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        K
    );

    cudaDeviceSynchronize();
    return C;
}
"""

matvec_mult_cpp_source = "torch::Tensor matvec_mult_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the custom CUDA kernel
matvec_mult = load_inline(
    name="matvec_mult",
    cpp_sources=matvec_mult_cpp_source,
    cuda_sources=matrix_vec_mult_source,
    functions=["matvec_mult_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matvec_mult = matvec_mult

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matvec_mult.matvec_mult_cuda(A.cuda(), B.cuda())