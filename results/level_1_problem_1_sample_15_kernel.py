import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
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
    auto threadsPerBlock = dim3(32, 8);
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    auto C = torch::empty({N, N}, A.options());

    matrixmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    cudaDeviceSynchronize();
    return C;
}
"""

matmul_header = "torch::Tensor matrixmul_cuda(torch::Tensor A, torch::Tensor B, int N);"

matmul_op = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_header,
    cuda_sources=matmul_source,
    functions=["matrixmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.N = 2048 * 2
        self.matmul_op = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_op.matrixmul_cuda(A, B, self.N)