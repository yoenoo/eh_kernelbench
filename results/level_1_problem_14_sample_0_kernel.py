import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for upper triangular matrix multiplication
triangular_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ 
void triu_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        if (row > col) return; // Output only upper triangular part

        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            // Exploit sparsity of upper triangular matrices
            // A[row][k] is zero when k < row
            if (k < row) continue;
            // B[k][col] is zero when col < k
            if (col < k) continue;
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor triu_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    const int block_size = 32;
    dim3 threads(block_size, block_size);
    dim3 blocks((N + block_size - 1)/block_size, (N + block_size - 1)/block_size);

    auto C = torch::empty({N, N}, A.options());

    triu_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    cudaDeviceSynchronize();

    return C;
}
"""

# Compile the inline CUDA code
tri_matmul = load_inline(
    name="tri_matmul",
    cuda_sources=triangular_matmul_source,
    functions=["triu_matmul_cuda"],
    verbose=True,
    extra_cflags=["-DLDAPре MLPре"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.triu_matmul = tri_matmul

    def forward(self, A, B):
        # Perform optimized triangular matrix multiplication
        return self.triu_matmul.triu_matmul_cuda(A, B)