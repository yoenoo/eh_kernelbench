import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

template <typename scalar_t>
__global__ void custom_matmul_transpose_kernel(const scalar_t* __restrict__ a, const scalar_t* __restrict__ b, scalar_t* __restrict__ c,
                                              int m, int k, int n) {
    extern __shared__ char scratch[];
    scalar_t* sub_a = reinterpret_cast<scalar_t*>(scratch);
    scalar_t* sub_b = reinterpret_cast<scalar_t*>(scratch) + (blockDim.x * (k/8));

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    scalar_t sum = 0.0;

    for (int block = 0; block < (k + blockDim.x - 1)/blockDim.x; ++block) {
        sub_a[ty * blockDim.x + tx] = a[row * k + block * blockDim.x + tx];
        sub_b[ty * blockDim.x + tx] = b[(block * blockDim.x + tx) * n + col];
        
        __syncthreads();

        for (int i = 0; i < blockDim.x; ++i) {
            sum += sub_a[ty * blockDim.x + i] * sub_b[i * blockDim.y + tx]; // fixed the index
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

torch::Tensor custom_matmul_transpose_cuda(torch::Tensor a, torch::Tensor b) {
    const int m = a.size(0);
    const int k = a.size(1);
    const int n = b.size(1);

    int block_size = 16;
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((n + dimBlock.x - 1)/dimBlock.x, (m + dimBlock.y - 1)/dimBlock.y);

    auto c = torch::empty({m, n}, a.options());

    const size_t shared_size = 2 * block_size * block_size * sizeof(float);
    custom_matmul_transpose_kernel<float><<<dimGrid, dimBlock, shared_size>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        m, k, n);
    
    return c;
}
"""

matmul_cpp = """
torch::Tensor custom_matmul_transpose_cuda(torch::Tensor a, torch::Tensor b);
"""

matmul_op = load_inline(
    name='custom_matmul',
    cpp_sources=matmul_cpp,
    cuda_sources=matmul_source,
    functions=['custom_matmul_transpose_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Our kernel expects the inputs in transposed form but we'll handle that here
        # Since original code was A.T @ B.T, equivalent to (B @ A)^T
        return self.matmul_op.custom_matmul_transpose_cuda(A.t(), B.t()).t()