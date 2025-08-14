import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batched matrix multiplication
batched_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void batched_matmul_kernel(
    const scalar_t* A,
    const scalar_t* B,
    scalar_t* C,
    int batch_size,
    int m,
    int k,
    int n
) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        scalar_t sum = 0;
        for (int e = 0; e < k; ++e) {
            sum += A[batch * m * k + row * k + e] * B[batch * k * n + e * n + col];
        }
        C[batch * m * n + row * n + col] = sum;
    }
}

std::tuple<int, int, int> get_grid_block_dims(
    int m,
    int n,
    int max_threads_per_block = 1024
) {
    int block_x = std::min(n, max_threads_per_block);
    int block_y = std::min(m, (max_threads_per_block + block_x - 1) / block_x);
    block_x = std::min(n, block_x);
    block_y = std::min(m, block_y);
    int block_size = block_x * block_y;

    dim3 block(block_x, block_y, 1);
    dim3 grid(
        (n + block_x - 1) / block_x,
        (m + block_y - 1) / block_y,
        batch_size
    );

    return std::make_tuple(grid, block, block_size);
}

torch::Tensor batched_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B,
    int batch_size,
    int m,
    int k,
    int n
) {
    auto C = torch::empty({batch_size, m, n}, A.options());

    auto grid_block = get_grid_block_dims(m, n);
    auto grid = std::get<0>(grid_block);
    auto block = std::get<1>(grid_block);

    AT_DISPATCH_ALL_TYPES(A.scalar_type(), "batched_matmul_cuda", ([&] {
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

batched_matmul_cpp_source = R"(
std::tuple<int, int, int> get_grid_block_dims(int m, int n, int max_threads_per_block = 1024);
at::Tensor batched_matmul_cuda(
    at::Tensor A,
    at::Tensor B,
    int batch_size,
    int m,
    int k,
    int n
);
)"

# Compile the inline CUDA code for batched matrix multiplication
batched_matmul = load_inline(
    name="batched_matmul",
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_source,
    functions=["batched_matmul_cuda", "get_grid_block_dims"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batch_size = 128
        self.m = 128 * 4
        self.k = 256 * 4
        self.n = 512 * 4
        self.batched_matmul_cuda = batched_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batched_matmul_cuda.batched_matmul_cuda(
            A,
            B,
            self.batch_size,
            self.m,
            self.k,
            self.n
        )