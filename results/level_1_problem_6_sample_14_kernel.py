import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

__global__ void custom_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* C, 
                                    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor custom_matmul(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1)/threads.x, (M + threads.y - 1)/threads.y);

    custom_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), 
                                             C.data_ptr<float>(), M, N, K);
    cudaDeviceSynchronize();
    return C;
}
"""

matmul_cpp_source = "torch::Tensor custom_matmul(torch::Tensor A, torch::Tensor B);"

matmul_extension = load_inline(
    name='custom_matmul',
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=['custom_matmul'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_extension

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.custom_matmul(A.cuda(), B.cuda())

# Note: Original get_inputs() will need to be adjusted to move tensors to CUDA