import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

# Define the custom CUDA kernel for matrix multiplication
matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_mult_kernel(const float* A, const float* B, float* C,
                                  int M, int K, int N) {
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

torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B) {
    const int threads_per_block = 32;
    dim3 blocks((N + threads_per_block - 1) / threads_per_block,
               (M + threads_per_block - 1) / threads_per_block, 1);
    dim3 threads(threads_per_block, threads_per_block);

    auto C = torch::empty({M, N}, A.options());

    matrix_mult_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N
    );

    cudaDeviceSynchronize();
    return C;
}
"""

matrix_mult_cpp_source = "torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matrix_mult = load_inline(
    name="matrix_mult",
    cpp_sources=matrix_mult_cpp_source,
    cuda_sources=matrix_mult_source,
    functions=["matrix_mult_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_mult = matrix_mult

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrix_mult.matrix_mult_cuda(A, B)

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []