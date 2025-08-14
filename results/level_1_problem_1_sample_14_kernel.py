import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C, 
                             int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        scalar_t sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int N) {
    auto C = torch::empty({N, N}, torch::dtype(A.dtype()).device(torch::kCUDA));

    const int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((N + block.x - 1)/block.x, (N + block.y - 1)/block.y);

    matmul_kernel<float><<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    cudaDeviceSynchronize();
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int N);
"""

matmul_op = load_inline(
    name="matmul_op",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_kernel_source,
    functions=["matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.N = 2048 * 2
        self.matmul_op = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_op.matmul_cuda(A.cuda(), B.cuda(), self.N)