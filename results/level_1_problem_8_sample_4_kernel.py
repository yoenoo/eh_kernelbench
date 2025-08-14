import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_irregular_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void matmul_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, 
                             int M, int K, int N) {
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

template <typename T>
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int M, int K, int N) {
    auto C = torch::empty({M, N}, A.options());

    const int block_x = 32;
    const int block_y = 8;
    dim3 block(block_x, block_y);
    dim3 grid((N + block_x - 1) / block_x, (M + block_y - 1) / block_y);

    matmul_kernel<T><<<grid, block>>>(A.data_ptr<T>(), B.data_ptr<T>(), C.data_ptr<T>(), M, K, N);
    cudaDeviceSynchronize();
    return C;
}

torch::Tensor matmul_float_cuda(torch::Tensor A, torch::Tensor B) {
    return matmul_cuda<float>(A, B, M, K, N);
}

torch::Tensor matmul_half_cuda(torch::Tensor A, torch::Tensor B) {
    return matmul_cuda<__half>(A, B, M, K, N);
}
"""

matmul_irregular_cpp = """
torch::Tensor matmul_float_cuda(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_half_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_irregular_ops = load_inline(
    name='matmul_irregular',
    cpp_sources=matmul_irregular_cpp,
    cuda_sources=matmul_irregular_source,
    functions=['matmul_float_cuda', 'matmul_half_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_float_cuda = matmul_irregular_ops.matmul_float_cuda
        self.matmul_half_cuda = matmul_irregular_ops.matmul_half_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.dtype == torch.float16:
            return self.matmul_half_cuda(A, B)
        else:
            return self.matmul_float_cuda(A, B)