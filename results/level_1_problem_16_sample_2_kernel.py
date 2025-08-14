import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrix_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template<typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int M, int K, int N) {
    const int threads = 32;
    dim3 blocks((N + threads - 1)/threads, (M + threads - 1)/threads);
    dim3 threadsPerBlock(threads, threads);

    auto C = torch::empty({M, N}, A.options());

    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_cuda", ([&] {
        matmul_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
            A.data<scalar_t>(), B.data<scalar_t>(), 
            C.data<scalar_t>(), M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}
"""

matrix_mul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int M, int K, int N);
"""

matmul_op = load_inline(
    name="custom_matmul",
    cpp_sources=matrix_mul_cpp_source,
    cuda_sources=microsecond,
    functions=["matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.M = 2048
        self.K = 8192
        self.N = 4096
        self.matmul_op = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure the inputs are on CUDA and in the right shape
        A = A.t().contiguous().cuda()
        B = B.contiguous().cuda()
        return self.matmul_op.matmul_cuda(A, B, self.M, self.K, self.N)