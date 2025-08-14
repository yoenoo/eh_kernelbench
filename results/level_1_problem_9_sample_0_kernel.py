import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* __restrict__ A,
                             const scalar_t* __restrict__ B,
                             scalar_t* __restrict__ C,
                             int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        scalar_t sum = 0;
        for (int e = 0; e < k; ++e) {
            sum += A[row * k + e] * B[e * n + col];
        }
        C[row * n + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int m = A.size(0);
    const int k = A.size(1);
    const int n = B.size(1);
    
    auto C = torch::empty({m, n}, A.options());

    int block_size_x = 32;
    int block_size_y = 8;
    dim3 block(block_size_x, block_size_y);
    
    int grid_x = (n + block_size_x - 1) / block_size_x;
    int grid_y = (m + block_size_y - 1) / block_size_y;
    dim3 grid(grid_x, grid_y);

    // Determine type and launch kernel
    AT_DISPATCH_ALL_TYPES(A.scalar_type(), "matmul_cuda", ([&] {
        matmul_kernel<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            m, n, k);
    }));
    
    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

matmul_op = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-DOPENVDB_ENABLE_KOKKOS=1",],
    extra_cuda_cflags=["-arch=sm_86"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul_op

    def forward(self, A, B):
        return self.matmul.matmul_cuda(A, B)