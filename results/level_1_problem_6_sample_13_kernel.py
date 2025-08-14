import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS 256

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        scalar_t sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(torch::kCUDA);
    torch::Tensor C = torch::empty({M, N}, options);

    dim3 threads(32, 8);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    matmul_kernel<float><<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    return C;
}
"""

matmul_kernel_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_op = load_inline(
    name="matmul_op",
    cpp_sources=matmul_kernel_cpp_source,
    cuda_sources=matmul_kernel_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_op.matmul_cuda(A.cuda(), B.cuda())

# Keep the same input dimensions as given
M = 256
N = 256
K = 131072 * 4

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []