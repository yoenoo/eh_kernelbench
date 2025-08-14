import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

diag_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void diag_matmul_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ out, const int N, const int M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        for (int col = 0; col < M; col++) {
            int index = row * M + col;
            out[index] = A[row] * B[index];
        }
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    const int M = B.size(1);
    auto out = torch::empty({N, M}, B.options());

    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(A.type(), "diag_matmul_cuda", ([&] {
        diag_matmul_kernel<scalar_t><<<grid_size, block_size>>>(
            A.data<scalar_t>(),
            B.data<scalar_t>(),
            out.data<scalar_t>(),
            N,
            M
        );
    }));

    return out;
}
"""

diag_matmul_header = "torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);"

diag_matmul = load_inline(
    name="diag_matmul",
    cpp_sources=diag_matmul_header,
    cuda_sources=diag_matmul_source,
    functions=["diag_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.diag_matmul = diag_matmul

    def forward(self, A, B):
        return self.diag_matmul.diag_matmul_cuda(A, B)