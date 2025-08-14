import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for square matrix multiplication (C = A * B)
matrixmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrixmul_kernel(const float* A, const float* B, float* C, int N) {
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

torch::Tensor matrixmul_cuda(torch::Tensor A, torch::Tensor B, int N) {
    auto output_options = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    auto C = torch::empty({N, N}, output_options);

    const int TPB = 32;
    dim3 threads(TPB, TPB);
    dim3 blocks((N + TPB - 1)/TPB, (N + TPB - 1)/TPB);

    matrixmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

matrixmul_cpp_source = "torch::Tensor matrixmul_cuda(torch::Tensor A, torch::Tensor B, int N);"

# Compile the inline CUDA code for matrix multiplication
matrixmul = load_inline(
    name="matrixmul",
    cpp_sources=matrixmul_cpp_source,
    cuda_sources=matrixmul_source,
    functions=["matrixmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 2048 * 2
        self.matrixmul = matrixmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure input tensors are on the same device and contiguous
        A = A.contiguous()
        B = B.contiguous()
        return self.matrixmul.matrixmul_cuda(A, B, self.N)