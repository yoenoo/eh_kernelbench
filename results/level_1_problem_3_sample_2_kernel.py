import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batched matrix multiplication
batched_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void batched_matmul_kernel(const scalar_t* A, const scalar_t* B, scalar_t* C,
                                     int batch_size, int m, int k, int n) {
    int batch = blockIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.z * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        scalar_t sum = 0;
        for (int e = 0; e < k; ++e) {
            sum += A[batch * m * k + row * k + e] * B[batch * k * n + e * n + col];
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

    dim3 threads(32, 8);  // 32 threads per block in x, 8 in y
    dim3 blocks(batch_size, (m + threads.y - 1) / threads.y, (n + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "batched_matmul_cuda", ([&] {
        batched_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data<scalar_t>(), B.data<scalar_t>(), C.data<scalar_t>(),
            batch_size, m, k, n);
    }));

    return C;
}
"""

batched_matmul_cpp_source = "torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
batched_matmul = load_inline(
    name="batched_matmul",
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_source,
    functions=["batched_matmul_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-arch=sm_80"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.batched_matmul = batched_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batched_matmul.batched_matmul_cuda(A, B)

def get_inputs():
    batch_size = 128
    m = 128 * 4
    k = 256 * 4
    n = 512 * 4
    A = torch.rand(batch_size, m, k).cuda()
    B = torch.rand(batch_size, k, n).cuda()
    return [A, B]

def get_init_inputs():
    return []