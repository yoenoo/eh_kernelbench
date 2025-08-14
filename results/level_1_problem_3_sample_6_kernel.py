import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

# Define the custom CUDA kernel for batched matrix multiplication
batch_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void batch_matmul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int batch_size,
    int m,
    int k,
    int n
) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        scalar_t sum = 0;
        for (int e = 0; e < k; ++e) {
            sum += A[batch_idx * m * k + row * k + e] * 
                   B[batch_idx * k * n + e * n + col];
        }
        C[batch_idx * m * n + row * n + col] = sum;
    }
}

torch::Tensor batch_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B
) {
    const int batch_size_ = A.size(0);
    const int m_ = A.size(1);
    const int k_ = A.size(2);
    const int n_ = B.size(2);

    auto C = torch::zeros({batch_size_, m_, n_}, A.options());

    int block_x = 32;
    int block_y = 8;
    dim3 block(block_x, block_y);
    dim3 grid(
        (n_ + block_x - 1) / block_x,
        (m_ + block_y - 1) / block_y,
        batch_size_
    );

    AT_DISPATCH_FLOATING_TYPES(A.type(), "batch_matmul_cuda", ([&] {
        batch_matmul_kernel<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            batch_size_,
            m_,
            k_,
            n_
        );
    }));

    return C;
}
"""

batch_matmul_cpp_source = R"""  
torch::Tensor batch_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
batch_matmul = load_inline(
    name="batch_matmul",
    cpp_sources=batch_matmul_cpp_source,
    cuda_sources=batch_matmul_source,
    functions=["batch_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batch_matmul = batch_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batch_matmul.batch_matmul_cuda(A, B)

def get_inputs():
    A = torch.rand(batch_size, m, k, device='cuda')
    B = torch.rand(batch_size, k, n, device='cuda')
    return [A, B]

def get_init_inputs():
    return []