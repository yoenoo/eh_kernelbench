import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for matrix multiplication (matmul)
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                             int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0;
        for (int e = 0; e < k; ++e) {
            sum += A[row * k + e] * B[e * n + col];
        }
        C[row * n + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);

    auto C = torch::empty({m, n}, torch::device("cuda"));

    dim3 threads(16, 16);
    dim3 blocks(
        (n + threads.x - 1) / threads.x,
        (m + threads.y - 1) / threads.y
    );

    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        m, k, n
    );

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

matmul = load_inline(
    name="matmul_op",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A.cuda(), B.cuda())