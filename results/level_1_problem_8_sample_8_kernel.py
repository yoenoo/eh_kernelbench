cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

M = 8205
K = 2949
N = 5921

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor custom_matmul(torch::Tensor A, torch::Tensor B) {
    const int threads_per_block = 32;
    dim3 blocks((N + threads_per_block - 1) / threads_per_block, (M + threads_per_block - 1) / threads_per_block, 1);
    dim3 threads(threads_per_block, threads_per_block);

    auto C = torch::empty({M, N}, A.options());

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    cudaDeviceSynchronize();

    return C;
}
"""

matmul_cpp_source = "torch::Tensor custom_matmul(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_op = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["custom_matmul"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.custom_matmul(A.cuda(), B.cuda())

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []