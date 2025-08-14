import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_tall_skinny_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// Define block size
#define BLOCK_SIZE 32

__global__ void matmul_tall_skinny_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* C,
                                      int m,
                                      int n,
                                      int k) {
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

torch::Tensor matmul_tall_skinny_cuda(torch::Tensor A, torch::Tensor B) {
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);

    auto C = torch::empty({m, n}, A.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1)/BLOCK_SIZE, (m + BLOCK_SIZE - 1)/BLOCK_SIZE);

    matmul_tall_skinny_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        m, n, k
    );

    return C;
}
"""

matmul_tall_skinny_cpp_source = "torch::Tensor matmul_tall_skinny_cuda(torch::Tensor A, torch::Tensor B);"

matmul_tall_skinny = load_inline(
    name="matmul_tall_skinny",
    cpp_sources=matmul_tall_skinny_cpp_source,
    cuda_sources=matmul_tall_skinny_source,
    functions=["matmul_tall_skinny_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_tall_skinny = matmul_tall_skinny

    def forward(self, A, B):
        return self.matmul_tall_skinny.matmul_tall_skinny_cuda(A, B)