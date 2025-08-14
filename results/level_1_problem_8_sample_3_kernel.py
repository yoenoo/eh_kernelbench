import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

template <typename scalar_t>
__global__ void fast_matmul_kernel(const scalar_t* __restrict__ A,
                                  const scalar_t* __restrict__ B,
                                  scalar_t* __restrict__ C,
                                  const int M,
                                  const int K,
                                  const int N) {
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

torch::Tensor fast_matmul(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    const int threads = 32;
    dim3 block(threads, threads);
    dim3 grid((N + threads - 1)/threads, (M + threads - 1)/threads);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "fast_matmul_cuda", ([&] {
        fast_matmul_kernel<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor fast_matmul(torch::Tensor A, torch::Tensor B);"
)

fast_matmul = load_inline(
    name="fast_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["fast_matmul"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = fast_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.fast_matmul(A.cuda(), B.cuda()).cpu() if A.is_cuda else self.matmul.fast_matmul(A, B)

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []