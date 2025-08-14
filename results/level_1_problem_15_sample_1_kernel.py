import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for lower triangular matrix multiplication
lt_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void lt_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N && row >= col) {
        float sum = 0.0;
        for (int k = 0; k <= col; ++k) { // B is lower triangular: B[k, col] is zero if k > col
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
        // Also set the element at (col, row) to zero except when row == col, but since we only compute lower triangle,
        // the upper triangle is not computed and remains zero. However, since the result is lower triangular, we can ignore
        // the upper part completely.
    }
}

torch::Tensor lt_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    const int threads = 32;
    dim3 blocks((N + threads - 1) / threads, (N + threads - 1) / threads);
    dim3 threadsPerBlock(threads, threads);

    auto C = torch::zeros({N, N}, A.options());

    lt_matmul_kernel<<<blocks, threadsPerBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    cudaDeviceSynchronize();
    return C;
}
"""

lt_matmul_cpp_source = "torch::Tensor lt_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
lt_matmul = load_inline(
    name="lt_matmul",
    cpp_sources=lt_matmul_cpp_source,
    cuda_sources=lt_matmul_source,
    functions=["lt_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.lt_matmul = lt_matmul

    def forward(self, A, B):
        return self.lt_matmul.lt_matmul_cuda(A, B)