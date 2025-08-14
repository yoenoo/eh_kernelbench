import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for upper triangular matrix multiplication
tri_upper_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void upper_triangular_matmul_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N || col >= N) return;

    scalar_t sum = 0;
    if (row <= col) {
        for (int k = row; k <= col; ++k) {
            sum += a[row * N + k] * b[k * N + col];
        }
    }
    c[row * N + col] = (row > col) ? 0 : sum;
}

std::tuple<torch::Tensor> upper_triangular_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    const int N = a.size(0);
    torch::Tensor c = torch::zeros({N, N}, a.options());

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y);

    upper_triangular_matmul_kernel<float>
        <<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N);

    cudaDeviceSynchronize();
    return c;
}
"""

# Compile the CUDA extension
tri_upper_matmul = load_inline(
    name="tri_upper_matmul",
    cpp_sources="",
    cuda_sources=tri_upper_matmul_source,
    functions=["upper_triangular_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tri_upper_matmul = tri_upper_matmul

    def forward(self, A, B):
        return self.tri_upper_matmul.upper_triangular_matmul_cuda(A, B)[0]