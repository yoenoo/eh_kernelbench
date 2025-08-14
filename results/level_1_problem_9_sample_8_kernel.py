import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>

template <typename T>
__global__ void tall_skinny_matmul(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, 
                                  int M, int K, int N) {
    constexpr int BLOCK_SIZE = 32;
    __shared__ T shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T shared_B[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = 0;

    for (int k = 0; k < K; k += BLOCK_SIZE) {
        // Load A tiles
        if (row < M && k + threadIdx.x < K) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0;
        }

        // Load B tiles
        if (col < N && k + threadIdx.y < K) {
            shared_B[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial sums
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += shared_A[threadIdx.y][i] * shared_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor tall_skinny_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(32, 32);
    dim3 blocks((N + threads.x - 1)/threads.x, (M + threads.y - 1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "tall_skinny_matmul_cuda", ([&] {
        tall_skinny_matmul<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}
"""

cpp_source = """
torch::Tensor tall_skinny_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_op = load_inline(
    name="tall_skinny_matmul",
    cpp_sources=cpp_source,
    cuda_sources=[matrix_mult_source],
    functions=["tall_skinny_matmul_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_op

    def forward(self, A, B):
        return self.matmul_op.tall_skinny_matmul_cuda(A, B)