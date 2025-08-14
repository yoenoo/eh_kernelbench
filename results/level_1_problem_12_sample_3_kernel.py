import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

diag_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        float sum = 0.0;
        // Since A is a diagonal matrix, only the diagonal elements contribute
        if (row == col) {
            sum = a[row] * b[row * M + col];
        }
        out[row * M + col] = sum;
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    int N = a.size(0);
    int M = b.size(1);
    auto out = torch::zeros({N, M}, a.options());

    dim3 threads(32, 8);
    dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    diag_matmul_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), N, M);
    cudaDeviceSynchronize();
    return out;
}
"""

diag_matmul_cpp_source = "torch::Tensor diag_matmul_cuda(torch::Tensor a, torch::Tensor b);"

diag_matmul = load_inline(
    name="diag_matmul",
    cuda_sources=diag_matmul_source,
    cpp_sources=diag_matmul_cpp_source,
    functions=["diag_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.diag_matmul = diag_matmul

    def forward(self, A, B):
        return self.diag_matmul.diag_matmul_cuda(A, B)