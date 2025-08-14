import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

diag_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(const float* A, const float* B, float* C, int N, int M) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < N && col < M) {
        C[row * M + col] = A[row] * B[row * M + col];
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = B.size(1);
    
    auto C = torch::empty({N, M}, B.options());
    
    dim3 threads(32, 8);
    dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    
    diag_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);
    
    return C;
}
"""

diag_matmul_cpp_source = "torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);"

diag_matmul = load_inline(
    name="diag_matmul",
    cpp_sources=diag_matmul_cpp_source,
    cuda_sources=diag_matmul_source,
    functions=["diag_matmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.diag_matmul = diag_matmul

    def forward(self, A, B):
        return self.diag_matmul.diag_matmul_cuda(A, B)