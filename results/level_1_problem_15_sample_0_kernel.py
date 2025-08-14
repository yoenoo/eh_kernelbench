import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for triangular matrix multiplication
tril_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void triangular_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k <= row; ++k) {
        if (k <= col) {
            sum += A[row * N + k] * B[k * N + col];
        }
    }
    C[row * N + col] = sum;
}

torch::Tensor tril_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y);

    auto C = torch::zeros({N, N}, A.options());

    triangular_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N
    );

    return C;
}
"""

tril_matmul_cpp_source = (
    "torch::Tensor tril_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the custom kernel
tril_matmul = load_inline(
    name="tril_matmul",
    cpp_sources=tril_matmul_cpp_source,
    cuda_sources=tril_matmul_source,
    functions=["tril_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tril_matmul = tril_matmul

    def forward(self, A, B):
        return self.tril_matmul.tril_matmul_cuda(A, B)