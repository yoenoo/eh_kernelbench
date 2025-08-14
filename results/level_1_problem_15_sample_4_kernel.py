import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for triangular matrix multiplication
tril_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void tril_matmul_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < N; k++) {
            // Since both A and B are lower triangular, A[row][k] and B[k][col] are zero when k > row or k > col respectively.
            // To leverage this sparsity:
            // For position (row, col), the valid range of k is max(0, col) <= k <= min(row, col)
            // But since both A and B are lower triangular:
            // A's row has non-zero elements from 0 to row
            // B's column (since B is transposed in multiplication? Wait, no. Let's think carefully.
            // Matrix multiplication: C[row][col] = sum_{k} A[row][k] * B[k][col]
            // Since B is lower triangular, B[k][col] is non-zero only when k >= col
            // Therefore, the non-zero terms are when k >= col and k <= row (since A is lower triangular)
            // So the valid k starts from col up to row, but also <= N-1
            if (k <= row && k >= col) {
                // Wait, this logic might be incorrect. Let me re-calculate
                // A[row][k] is non-zero only when k <= row (since A is lower triangular)
                // B[k][col] is non-zero only when k >= col (since B is lower triangular)
                // Therefore, the product is non-zero when col <= k <= row
                sum += A[row * N + k] * B[k * N + col];
            }
        }
        if (row >= col) {  // The result should only keep the lower triangular part
            C[row * N + col] = sum;
        } else {
            C[row * N + col] = 0;
        }
    }
}

torch::Tensor tril_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y);

    // Launch the kernel with appropriate grid and block dimensions
    tril_matmul_kernel<float><<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

tril_matmul_cpp_source = "torch::Tensor tril_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
tril_matmul = load_inline(
    name="tril_matmul",
    cpp_sources=tril_matmul_cpp_source,
    cuda_sources=tril_matmul_source,
    functions=["tril_matmul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tril_matmul = tril_matmul

    def forward(self, A, B):
        return self.tril_matmul.tril_matmul_cuda(A, B)