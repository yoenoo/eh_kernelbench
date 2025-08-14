cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* __restrict__ a,
                             const scalar_t* __restrict__ b,
                             scalar_t* __restrict__ c,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

at::Tensor matmul_cuda(at::Tensor a, at::Tensor b, int M, int N, int K) {
    const int threads = 32;
    dim3 block(threads, threads);
    int grid_x = (N + threads - 1) / threads;
    int grid_y = (M + threads - 1) / threads;
    dim3 grid(grid_x, grid_y);

    at::Tensor c = at::empty({M, N}, a.options());

    matmul_kernel<float><<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        M, N, K);

    return c;
}
"""

matmul_cpp_source = R"""
at::Tensor matmul_cuda(at::Tensor a, at::Tensor b, int M, int N, int K);
"""

matmul = load_inline(
    name="matmul_op",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.M = 16384 * 2
        self.N = 16384 * 2
        self.K = 32 * 2
        self.matmul = matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A, B, self.M, self.N, self.K)