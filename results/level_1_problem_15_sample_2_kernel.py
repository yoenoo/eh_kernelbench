import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for triangular matrix multiplication
triangular_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void tri_matmul_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        T sum = 0;
        for (int k = 0; k <= min(row, col); ++k) {
            int A_idx = row * N + k;
            int B_idx = k * N + col;
            sum += static_cast<double>(A[A_idx]) * static_cast<double>(B[B_idx]);
        }
        C[row * N + col] = static_cast<T>(sum);
    }
}

torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "triangular_matmul_cuda", ([&] {
        tri_matmul_kernel<scalar_t><<<blocks, threads>>>(A.data<scalar_t>(), B.data<scalar_t>(), C.data<scalar_t>(), N);
    }));

    return C;
}
"""

triangular_matmul_header = """
torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

triangular_matmul = load_inline(
    name="triangular_matmul",
    cpp_sources=triangular_matmul_header,
    cuda_sources=triangular_matmul_source,
    functions=["triangular_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.triangular_matmul = triangular_matmul

    def forward(self, A, B):
        C = self.triangular_matmul.triangular_matmul_cuda(A, B)
        return C