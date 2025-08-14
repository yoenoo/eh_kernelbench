import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication with transpose
matmul_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void matmul_transpose_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int m, int k, int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        scalar_t sum = 0;
        for (int e = 0; e < k; ++e) {
            sum += a[row + e * m] * b[e * n + col];
        }
        c[row * n + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor a, torch::Tensor torch::Tensor b) {
    const int m = a.size(0);
    const int k = a.size(1);
    const int n = b.size(1);

    auto c = torch::empty({m, n}, a.options());

    const int threads = 32;
    dim3 blocks(TORCH_GPU_BLOCK(!, (n + threads - 1) / threads), 
               TORCH_GPU_BLOCK(!, (m + threads - 1) / threads));
    dim3 threads(threads, threads);

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "matmul_transpose_cuda", ([&] {
        matmul_transpose_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(),
            m, k, n);
    }));

    return c;
}
"""

matmul_transpose_cpp_source = """
torch::Tensor matmul_transpose_cuda(torch::Tensor a, torch::Tensor b);
"""

# Compile the inline CUDA code for matrix multiplication with transpose
matmul_transpose = load_inline(
    name="matmul_transpose",
    cpp_sources=matmul_transpose_cpp_source,
    cuda_sources=matmul_transpose_source,
    functions=["matmul_transpose_cuda"],
    verbose=True,
    extra_cflags=["-D_DEBUG"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_transpose = matmul_transpose

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.t()
        return self.matmul_transpose.matmul_transpose_cuda(A, B)

def get_inputs():
    K = 4096 * 2
    M = 1024 * 2
    N = 2048 * 2
    A = torch.randn(K, M).cuda()
    B = torch.randn(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []