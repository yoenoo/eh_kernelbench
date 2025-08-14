import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_large_K_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_large_K_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    const dim3 threads(16, 16);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K
    );

    cudaDeviceSynchronize();
    return C;
}
"""

matmul_large_K_header = "torch::Tensor matmul_large_K_cuda(torch::Tensor A, torch::Tensor B);"

matmul_large_K = load_inline(
    name="matmul_large_K",
    cpp_sources=matmul_large_K_header,
    cuda_sources=matmul_large_K_source,
    functions=["matmul_large_K_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_large_K = matmul_large_K

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_large_K.matmul_large_K_cuda(A, B)

M = 256
N = 256
K = 131072 * 4

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []