cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_large_k_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_large_k_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_large_k_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    const int threads = 32;
    dim3 blocks(TORCH_DIVIDE_FRIST(N, threads), TORCH_DIVIDE_FRIST(M, threads));
    dim3 threadsPerBlock(threads, threads);

    matmul_large_k_kernel<float><<<blocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K);

    return C;
}
"""

matmul_large_k_cpp_source = "torch::Tensor matmul_large_k_cuda(torch::Tensor A, torch::Tensor B);";

matmul_large_k = load_inline(
    name="matmul_large_k",
    cpp_sources=matmul_large_k_cpp_source,
    cuda_sources=matmul_large_k_source,
    functions=["matmul_large_k_cuda"],
    verbose=True,
    extra_cflags=["-D稀释=30"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_large_k = matmul_large_k

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_large_k.matmul_large_k_cuda(A.cuda(), B.cuda())

M = 256
N = 256
K = 131072 * 4

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []