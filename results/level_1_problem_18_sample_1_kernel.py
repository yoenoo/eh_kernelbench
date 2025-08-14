import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication (C = A^T * B^T)
matmul_transposed_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void matmul_transposed_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    const int m,
    const int k,
    const int n,
    const int lda,
    const int ldb,
    const int ldc) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bz = blockIdx.z;

    __shared__ scalar_t shared_a[32][32];
    __shared__ scalar_t shared_b[32][32];

    scalar_t acc = 0.0;
    for (int i = 0; i < (k - 1) / 32 + 1; ++i) {
        // Load A tile into shared memory
        const int a_row = ty + i * 32 * 2 + bz * 32 * 2; // Adjust for batch (blocking)
        const int a_col = tx;
        const int a_idx = a_row * lda + a_col;
        shared_a[ty][tx] = (a_idx < m * k) ? a[a_idx] : 0.0;
        
        // Load B tile into shared memory
        const int b_row = tx;
        const int b_col = ty + i * 32 * 2 + bz * 32 * 2;
        const int b_idx = b_row * ldb + b_col;
        shared_b[ty][tx] = (b_idx < k * n) ? b[b_idx] : 0.0;

        __syncthreads();

        // Compute the dot product of the current tiles
        for (int j = 0; j < 32; ++j) {
            acc += shared_a[ty][j] * shared_b[j][tx];
        }

        __syncthreads();
    }
    c[ty * ldc + tx + bz * 32 * blockDim.x] = acc;
}

at::Tensor matmul_transposed_cuda(
    at::Tensor a, 
    at::Tensor b,
    int m, 
    int k,
    int n) {

    const int block_size = 32;
    const dim3 block(block_size, block_size);
    const dim3 grid(1, 1, (k + 32 - 1)/32);

    auto c = at::empty({m, n}, a.options());

    const int lda = a.size(1);
    const int ldb = b.size(1);
    const int ldc = c.size(1);

    matmul_transposed_kernel<float><<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        m,
        k,
        n,
        lda,
        ldb,
        ldc);

    return c;
}
"""

matmul_transposed_cpp_source = (
    "at::Tensor matmul_transposed_cuda(at::Tensor a, at::Tensor b, int m, int k, int n);"
)

# Compile the inline CUDA code for transposed matrix multiplication
matmul_transposed = load_inline(
    name="matmul_transposed",
    cpp_sources=matmul_transposed_cpp_source,
    cuda_sources=matmul_transposed_source,
    functions=["matmul_transposed_cuda"],
    verbose=True,
    extra_cflags=["-arch=sm_80"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_transposed = matmul_transposed
        self.m = M
        self.k = K
        self.n = N

    def forward(self, A, B):
        # Transpose the inputs as per the original model's requirements (since torch.matmul(A.T, B.T))
        A = A.t().contiguous() # Explicit memory layout optimization
        B = B.t().contiguous()
        return self.matmul_transposed.matmul_transposed_cuda(A, B, self.m, self.k, self.n)