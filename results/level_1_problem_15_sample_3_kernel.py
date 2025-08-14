import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for triangular matrix multiplication
tri_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename T>
__global__ void tri_matmul_kernel(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        if (row >= col) {
            T sum = 0;
            for (int k = 0; k <= col; ++k) {
                sum += a[row * N + k] * b[k * N + col];
            }
            c[row * N + col] = sum;
        }
    }
}

torch::Tensor tri_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    const int N = a.size(0);
    auto c = torch::zeros_like(a);

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "tri_matmul_cuda", ([&] {
        tri_matmul_kernel<scalar_t><<<blocks, threads>>>(a.data_ptr<scalar_type>(), b.data_ptr<scalar_type>(), c.data_ptr<scalar_type>(), N);
    }));

    return c;
}
"""

tri_matmul_cpp_source = "torch::Tensor tri_matmul_cuda(torch::Tensor a, torch::Tensor b);"

# Compile the custom CUDA operator
tri_matmul = load_inline(
    name="tri_matmul",
    cpp_sources=tri_matmul_cpp_source,
    cuda_sources=tri_matmul_source,
    functions=["tri_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.tri_matmul = tri_matmul

    def forward(self, A, B):
        return self.tri_matmul.tri_matmul_cuda(A, B)