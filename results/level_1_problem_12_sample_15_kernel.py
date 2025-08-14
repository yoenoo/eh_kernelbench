import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

diag_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void diag_matmul_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ out, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < M) {
        out[row * M + col] = A[row] * B[row * M + col];
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = B.size(1);
    auto out = torch::empty({N, M}, B.options());

    dim3 threads(32, 8);
    dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "diag_matmul_cuda", ([&] {
        diag_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data<scalar_t>(), B.data<scalar_t>(), out.data<scalar_t>(), N, M);
    }));

    return out;
}
"""

diag_matmul_cpp_source = (
    "torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

diag_matmul = load_inline(
    name="diag_matmul",
    cpp_sources=diag_matmul_cpp_source,
    cuda_sources=diag_matmul_source,
    functions=["diag_matmul_cuda"],
    verbose=True,
    extra_cflags=["-DVERSION_GE_1_5"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.diag_matmul = diag_matmul

    def forward(self, A, B):
        return self.diag_matmul.diag_matmul_cuda(A, B)