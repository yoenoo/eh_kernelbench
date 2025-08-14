import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for upper triangular matrix multiplication
matmul_upper_triangular_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_upper_triangular_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    // Since matrices are upper triangular, C[row][col] is zero if row > col
    if (row > col) return;

    scalar_t sum = 0;
    for (int k = row; k < N; ++k) {  // A is upper triangular, so rows before row are zero
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

torch::Tensor matmul_upper_triangular_cuda(torch::Tensor A, torch::Tensor B, int N) {
    auto device = A.device();
    auto C = torch::empty({N, N}, torch::device(device).dtype(torch::kFloat32));

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_upper_triangular_cuda", ([&] {
        matmul_upper_triangular_kernel<scalar_t>
            <<<blocks, threads>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                N);
    }));

    return C;
}
"""

matmul_upper_triangular_cpp_source = (
    "torch::Tensor matmul_upper_triangular_cuda(torch::Tensor A, torch::Tensor B, int N);"
)

# Compile the inline CUDA code
matmul_upper_tri = load_inline(
    name="matmul_upper_tri",
    cpp_sources=matmul_upper_triangular_cpp_source,
    cuda_sources=matmul_upper_triangular_source,
    functions=["matmul_upper_triangular_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 4096
        self.matmul_upper_tri = matmul_upper_tri

    def forward(self, A, B):
        return self.matmul_upper_tri.matmul_upper_triangular_cuda(A, B, self.N)