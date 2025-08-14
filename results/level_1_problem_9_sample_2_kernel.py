import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication optimized for tall/skinny matrices
matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename T>
__global__ void matmul_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
                             int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        T sum = 0;
        for (int e = 0; e < k; e++) {
            sum += A[row * k + e] * B[e * n + col];
        }
        C[row * n + col] = sum;
    }
}

torch::Tensor matmul_tall_skinny_cuda(torch::Tensor A, torch::Tensor B) {
    const int m = A.size(0);
    const int k = A.size(1);
    const int n = B.size(1);

    auto C = torch::empty({m, n}, A.options());

    const int threads = 32;
    dim3 blocks((n + threads - 1)/threads, (m + threads - 1)/threads);
    dim3 threads_block(threads, threads);

    AT_DISPATCH_ALL_TYPES(A.scalar_type(), "matmul_tall_skinny_cuda", ([&] {
        matmul_kernel<scalar_t><<<blocks, threads_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            m, n, k
        );
    }));

    cudaDeviceSynchronize();
    return C;
}
"""

matmul_kernel_cpp = "torch::Tensor matmul_tall_skinny_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the custom CUDA kernel
matmul_op = load_inline(
    name="matmul_tall_skinny",
    cpp_sources=[matmul_kernel_cpp],
    cuda_sources=[matmul_kernel_source],
    functions="matmul_tall_skinny_cuda",
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_op

    def forward(self, A, B):
        return self.matmul.matmul_tall_skinny_cuda(A, B)