import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* C,
                             int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ scalar_t s_A[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t s_B[TILE_SIZE][TILE_SIZE];

    scalar_t sum = 0.0;

    for (int t = 0; t < (K - 1) / TILE_SIZE + 1; ++t) {
        // Load tiles of A and B into shared memory
        int a_row = by * TILE_SIZE + ty;
        int a_col = t * TILE_SIZE + tx;
        s_A[ty][tx] = (a_col < K) ? A[a_row * K + a_col] : 0.0;

        int b_row = bx * TILE_SIZE + tx;
        int b_col = t * TILE_SIZE + ty;
        s_B[ty][tx] = (b_col < K) ? B[b_col * N + b_row] : 0.0; // B is transposed here (since B.T is used)

        __syncthreads();

        // Compute the dot product of the tile fragments
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }

    // Write the computed value to the output matrix
    int c_row = by * TILE_SIZE + ty;
    int c_col = bx * TILE_SIZE + tx;
    if (c_row < M && c_col < N) {
        C[c_row * N + c_col] = sum;
    }
}

#define TILE_SIZE 32

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0); // Since B is transposed, original B's rows are columns now.

    auto C = torch::empty({M, N}, A.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch the kernel
    matmul_kernel<float><<<grid, block, 0, stream>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    cudaDeviceSynchronize();
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cuda_cflags=["-arch=sm_75"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A, B)