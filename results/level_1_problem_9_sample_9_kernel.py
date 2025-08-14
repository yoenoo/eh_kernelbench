import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication optimized for tall/skinny matrices
matmul_tall_skinny_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void matmul_tall_skinny_kernel(const scalar_t* __restrict__ A,
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

torch::Tensor matmul_tall_skinny_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure input dimensions are compatible
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);
    assert(k == B.size(0), "Incompatible matrix dimensions.");

    // Create output tensor
    auto C = torch::empty({m, n}, A.options());

    const int threads = 32;
    dim3 block(threads, threads);
    dim3 grid((n + threads - 1) / threads, (m + threads - 1) / threads);

    // Launch kernel with proper template based on data type
    AT_DISPATCH_ALL_TYPES(A.scalar_type(), "matmul_tall_skinny_cuda", ([&] {
        matmul_tall_skinny_kernel<scalar_t><<<grid, block>>>(
            A.data<scalar_t>(),
            B.data<scalar_t>(),
            C.data<scalar_t>(),
            m, n, k);
    }));

    cudaDeviceSynchronize();
    return C;
}
"""

matmul_tall_skinny_cpp_source = "torch::Tensor matmul_tall_skinny_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the custom CUDA kernel
matmul_tall_skinny = load_inline(
    name="matmul_tall_skinny",
    cpp_sources=matmul_tall_skinny_cpp_source,
    cuda_sources=matmul_tall_skinny_source,
    functions=["matmul_tall_skinny_cuda"],
    verbose=True,
    extra_cflags=["-D_FORCE_INLINES", "-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_tall_skinny = matmul_tall_skinny

    def forward(self, A, B):
        return self.matmul_tall_skinny.matmul_tall_skinny_cuda(A, B)

def get_inputs():
    M = 16384 * 2
    N = 16 * 2
    A = torch.rand(M, N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]

def get_init_inputs():
    return []