import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fast_matmul_kernel(
    scalar_t* __restrict__ a,
    scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        scalar_t sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += a[row * lda + i] * b[i * ldb + col];
        }
        c[row * ldc + col] = sum;
    }
}

torch::Tensor fast_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    auto c = torch::zeros({M, N}, a.options());

    const int threads = 32;
    dim3 blocks((N + threads - 1) / threads, (M + threads - 1) / threads);

    const int lda = a.stride(0);
    const int ldb = b.stride(0);
    const int ldc = c.stride(0);

    dim3 threadsPerBlock(threads, threads);

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "fast_matmul_cuda", ([&]{
        fast_matmul_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
            a.data<scalar_t>(),
            b.data<scalar_t>(),
            c.data<scalar_t>(),
            M, N, K,
            lda, ldb, ldc
        );
    }));

    return c;
}
"""

matmul_kernel_header = """
torch::Tensor fast_matmul_cuda(torch::Tensor a, torch::Tensor b);
"""

matmul_op = load_inline(
    name='fast_matmul',
    cpp_sources=matmul_kernel_header,
    cuda_sources=matmul_kernel_source,
    functions=['fast_matmul_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.fast_matmul_cuda(A.t(), B.t())

# Original input generator
def get_inputs():
    A = torch.rand(K, M).cuda()
    B = torch.rand(N, K).cuda()
    return [A, B]

def get_init_inputs():
    return []