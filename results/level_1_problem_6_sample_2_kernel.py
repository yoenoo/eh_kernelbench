import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

template <typename T>
__global__ void custom_matmul(const T* __restrict__ A, const T* __restrict__ B, T* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += __ldg(A + row * K + k) * __ldg(B + k * N + col);
        }
        C[row * N + col] = sum;
    }
}

#define DECLARE_KERNEL(T) \\
    template __global__ void custom_matmul<T><<<dimGrid, dimBlock>>>(const T*, const T*, T*, int, int, int);

DECLARE_KERNEL(float)
DECLARE_KERNEL(at::Half)

torch::Tensor custom_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());

    int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((N + block_size - 1)/block_size, (M + block_size - 1)/block_size);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "custom_matmul_cuda", ([&] {
        using T = scalar_t;
        custom_matmul<T><<<grid, block>>>(
            A.data_ptr<T>(), B.data_ptr<T>(), 
            C.data_ptr<T>(), M, N, K);
    }));

    cudaDeviceSynchronize();
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor custom_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cu = load_inline(
    name='custom_matmul_cuda',
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_kernel_source,
    functions=['custom_matmul_cuda'],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cuda_matmul = matmul_cu

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.cuda_matmul.custom_matmul_cuda(A.cuda(), B.cuda())