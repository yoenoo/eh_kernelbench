import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((N + block_size - 1)/block_size, (N + block_size - 1)/block_size);

    auto C = torch::empty({N, N}, A.options());

    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    
    cudaDeviceSynchronize();
    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

matmul_op = load_inline(
    name="matmul_op",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matrix_mult_source,
    functions=["matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda.matmul_cuda(A.cuda(), B.cuda())