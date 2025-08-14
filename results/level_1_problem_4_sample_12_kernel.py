import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_vector_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void matmul_vector_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C,
                                    int M, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;

    scalar_t sum = 0;
    for (int k = 0; k < K; ++k) {
        sum += A[idx * K + k] * B[k];
    }
    C[idx] = sum;
}

torch::Tensor matmul_vector_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    auto C = torch::zeros({M, 1}, A.options());

    const int block_size = 256;
    const int grid_size = (M + block_size - 1) / block_size;

    // Use float for computation precision
    matmul_vector_kernel<float><<<grid_size, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K);

    return C;
}
"""

matmul_vector_cpp_source = "torch::Tensor matmul_vector_cuda(torch::Tensor A, torch::Tensor B);"

matmul_vector = load_inline(
    name="matmul_vector",
    cpp_sources=matmul_vector_cpp_source,
    cuda_sources=matmul_vector_source,
    functions=["matmul_vector_cuda"],
    verbose=False,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-lineinfo", "-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_vector = matmul_vector

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_vector.matmul_vector_cuda(A, B)

# Keep these definitions same as original for compatibility
M = 256 * 8
K = 131072 * 8

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, 1).cuda()
    return [A, B]

def get_init_inputs():
    return []