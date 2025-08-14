import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for upper triangular matrix multiplication
matmul_triangular_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/Parallel.h>

template <typename scalar_t>
__global__ void matmul_triangular_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    scalar_t sum = 0;
    for (int k = 0; k < N; k++) {
        // Exploit upper triangular property to skip lower triangle elements
        if (k > row || k > col) continue;
        sum += A[row * N + k] * B[k * N + col];
    }
    if (row <= col) {  // Only store upper triangular part
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_triangular_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    matmul_triangular_kernel<float><<<blocks, threads, 0, stream>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}
"""

matmul_triangular_cpp_source = (
    "torch::Tensor matmul_triangular_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
matmul_triangular = load_inline(
    name="matmul_triangular",
    cpp_sources=matmul_triangular_cpp_source,
    cuda_sources=matmul_triangular_source,
    functions=["matmul_triangular_cuda"],
    verbose=True,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_triangular = matmul_triangular

    def forward(self, A, B):
        return self.matmul_triangular.matmul_triangular_cuda(A, B)