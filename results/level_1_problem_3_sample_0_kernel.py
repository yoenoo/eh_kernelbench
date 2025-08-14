import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batched matrix multiplication
batched_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/macros/Macros.h>

template <typename scalar_t>
__global__ void batched_matmul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int batch_size,
    int m,
    int k,
    int n
) {
    int batch = blockIdx.x;
    int col = threadIdx.x + blockIdx.y * blockDim.x;
    int row = threadIdx.y + blockIdx.z * blockDim.y;

    if (row < m && col < n) {
        scalar_t sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += A[batch * m * k + row * k + i] * B[batch * k * n + i * n + col];
        }
        C[batch * m * n + row * n + col] = sum;
    }
}

torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int batch_size = A.size(0);
    const int m = A.size(1);
    const int k = A.size(2);
    const int n = B.size(2);

    auto C = torch::empty({batch_size, m, n}, A.options());

    const int block_dim_x = 32;
    const int block_dim_y = 8;
    dim3 block(block_dim_x, block_dim_y);
    
    dim3 grid(
        batch_size,
        (n + block_dim_x - 1) / block_dim_x,
        (m + block_dim_y - 1) / block_dim_y
    );

    AT_DISPATCH_FLOATING_TYPES(A.type(), "batched_matmul_cuda", ([&] {
        batched_matmul_kernel<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            batch_size,
            m,
            k,
            n
        );
    }));

    return C;
}
"""

batched_matmul_cpp_source = """
torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code for batched matrix multiplication
batched_matmul = load_inline(
    name="batched_matmul",
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_source,
    functions=["batched_matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_matmul = batched_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batched_matmul.batched_matmul_cuda(A, B)