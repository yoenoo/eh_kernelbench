import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrix_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matrix_mul_kernel(const scalar_t* __restrict__ A,
                                 const scalar_t* __restrict__ B,
                                 scalar_t* __restrict__ C,
                                 const int M, const int K, const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matrix_mul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    const int threads = 32;
    dim3 blocks(T ceil(1.0 * N / threads), T ceil(1.0 * M / threads));

    AT_DISPATCH_FLOATING_TYPES(A.type(), "matrix_mul_cuda", ([&] {
        matrix_mul_kernel<scalar_t><<<blocks, threads>>>(
            A.data<scalar_t>(), B.data<scalar_t>(), C.data<scalar_t>(),
            M, K, N);
    }));

    return C;
}
"""

matrix_mul_cpp_source = "torch::Tensor matrix_mul_cuda(torch::Tensor A, torch::Tensor B);"

matrix_mul = load_inline(
    name="matrix_mul",
    cpp_sources=matrix_mul_cpp_source,
    cuda_sources=matrix_mul_source,
    functions=["matrix_mul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_mul = matrix_mul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Transpose A before passing to CUDA kernel
        return self.matrix_mul.matrix_mul_cuda(A.t(), B)