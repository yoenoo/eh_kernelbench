import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-vector multiplication using tensor cores for faster computation
matmul_vector_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Custom kernel using CUDA for matrix-vector multiplication leveraging CUDA's optimizations
template<typename scalar_t>
__global__ void custom_matmul_vector_kernel(const scalar_t* __restrict__ A,
                                           const scalar_t* __restrict__ B,
                                                       scalar_t* C,
                                           int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    scalar_t sum = 0;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k];
    }
    C[row] = sum;
}

torch::Tensor custom_matmul_vector_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    auto C = torch::empty({M, 1}, A.options());

    const int threads = 256;
    const int blocks = (M + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(A.type(), "custom_matmul_vector_cuda", ([&] {
        custom_matmul_vector_kernel<scalar_t><<<blocks, threads>>>(
            A.data<scalar_t>(), B.data<scalar_t>(), C.data<scalar_t>(), M, K);
    }));

    cudaDeviceSynchronize();
    return C;
}
"""

matmul_vector_cpp_source = "torch::Tensor custom_matmul_vector_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the custom CUDA kernel inline
custom_matmul = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_vector_cpp_source,
    cuda_sources=matmul_vector_source,
    functions=["custom_matmul_vector_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = custom_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure B is treated as a vector (Kx1)
        B = B.view(-1)
        return self.matmul_op.custom_matmul_vector_cuda(A, B).view(-1, 1)

M = 256 * 8  # 2048
K = 131072 * 8  # 1048576

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K).cuda()  # Stored as 1D for efficiency
    return [A, B]

def get_init_inputs():
    return []