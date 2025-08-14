import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
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

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1)/threads.x, (M + threads.y - 1)/threads.y);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    cudaDeviceSynchronize();
    return C;
}
"""

matmul_kernel_header = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda_op = load_inline(
    name='matmul_cuda',
    cpp_sources=matmul_kernel_header,
    cuda_sources=matmul_kernel_source,
    functions='matmul_cuda',
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda_op = matmul_cuda_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda_op.matmul_cuda(A.T, B)