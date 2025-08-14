import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_diag_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(const float* A, const float* B, float* C, int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * M) {
        return;
    }
    int i = idx / M;
    C[idx] = A[i] * B[idx];
}

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Validate input dimensions
    int64_t N = A.sizes()[0];
    int64_t M = B.sizes()[1];
    TORCH_CHECK(A.dim() == 1, "A must be 1D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch between A and B");

    auto C = torch::empty_like(B);
    
    const int block_size = 256;
    int total_elements = N * M;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    diag_matmul_kernel<<<num_blocks, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}
"""

elementwise_diag_matmul_cpp = "torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);"

diag_matmul = load_inline(
    name="diag_matmul",
    cpp_sources=elementwise_diag_matmul_cpp,
    cuda_sources=elementwise_diag_matmul_source,
    functions=["diag_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.diag_matmul = diag_matmul

    def forward(self, A, B):
        return self.diag_matmul.diag_matmul_cuda(A, B)