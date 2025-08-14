import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batched matrix multiplication
batched_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void batched_matmul_kernel(const scalar_t* A, const scalar_t* B, scalar_t* C,
                                     int N, int M, int K, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * M) {
        return;
    }
    int n = idx / M;
    int m = idx % M;
    for (int l = 0; l < L; l++) {
        scalar_t sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[n * M * K + m * K + k] * B[k * L + l];
        }
        C[n * M * L + m * L + l] = sum;
    }
}

torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    auto C = torch::empty({N, M, L}, A.options());

    const int block_size = 256;
    const int num_elements = N * M;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(A.type(), "batched_matmul_cuda", ([&] {
        batched_matmul_kernel<scalar_t><<<num_blocks, block_size>>>(
            A.data<scalar_t>(), B.data<scalar_t>(), C.data<scalar_t>(),
            N, M, K, L);
    }));

    return C;
}
"""

batched_matmul_cpp_source = (
    "torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
batched_matmul = load_inline(
    name="batched_matmul",
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_source,
    functions=["batched_matmul_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.batched_matmul = batched_matmul

    def forward(self, A, B):
        return self.batched_matmul.batched_matmul_cuda(A, B)