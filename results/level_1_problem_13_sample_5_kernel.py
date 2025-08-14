import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void sym_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
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

torch::Tensor sym_matmul_cuda(torch::Tensor A, torch::Tensor B, int N) {
    const int threads = BLOCK_SIZE;
    dim3 blocks((N + threads - 1) / threads, (N + threads - 1) / threads);

    auto C = torch::zeros({N, N}, A.options());

    sym_matmul_kernel<<<blocks, dim3(threads, threads)>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );
    cudaDeviceSynchronize();

    return C;
}
"""

cpp_source = """
torch::Tensor sym_matmul_cuda(torch::Tensor A, torch::Tensor B, int N);
"""

sym_matmul = load_inline(
    name="sym_matmul",
    cpp_sources=cpp_source,
    cuda_sources=kernel_source,
    functions=["sym_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.N = 4096
        self.sym_matmul = sym_matmul

    def forward(self, A, B):
        return self.sym_matmul.sym_matmul_cuda(A, B, self.N)