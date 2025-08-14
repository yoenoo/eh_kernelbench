import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void matmul_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
                             int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; k++) {
            sum += static_cast<T>(A[row * K + k]) * static_cast<T>(B[k * N + col]);
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    const dim3 threads(32, 8);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    matmul_kernel<float><<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N
    );

    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_op = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_kernel_source,
    functions=["matmul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_op.matmul_cuda(A, B)

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K, device='cuda')
    B = torch.rand(K, N, device='cuda')
    return [A, B]

def get_init_inputs():
    return []