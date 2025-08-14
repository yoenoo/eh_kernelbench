import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    const int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((N + block_size - 1) / block_size, (M + block_size - 1) / block_size);

    // Selecting the kernel type based on data type
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_cuda", ([&] {
        matmul_kernel<scalar_t><<<grid, block>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(), M, N, K);
    }));

    return C;
}
"""

matrix_mult_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

matmul_op = load_inline(
    name="matmul_op",
    cpp_sources=matrix_mult_cpp_source,
    cuda_sources=matrix_mult_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A, B)