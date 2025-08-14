import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrix_multiplication_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <ATen/cuda/CUDAContext.h>

template <typename T>
__global__ void fast_matmul_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int m,
    int k,
    int n,
    int lda,
    int ldb,
    int ldc
) {
    __shared__ T shared_a[32][32];
    __shared__ T shared_b[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    T acc = static_cast<T>(0);

    for (int t = 0; t < (k + 31) / 32; ++t) {
        // Load tiles of a and b into shared memory
        int a_row = block_row * 32 + ty;
        int a_col = t * 32 + tx;
        shared_a[ty][tx] = (a_col < k) ? a[a_row * lda + a_col] : 0;

        int b_row = t * 32 + ty;
        int b_col = block_col * 32 + tx;
        shared_b[ty][tx] = (b_row < k) ? b[b_row * ldb + b_col] : 0;

        __syncthreads();

        // Perform the matrix multiplication on the shared memory tiles
        for (int i = 0; i < 32; ++i) {
            acc += shared_a[ty][i] * shared_b[i][tx];
        }

        __syncthreads();
    }

    int row = block_row * 32 + ty;
    int col = block_col * 32 + tx;

    if (row < m && col < n) {
        c[row * ldc + col] = acc;
    }
}

torch::Tensor fast_matmul_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::ScalarType dtype = torch::kFloat32
) {
    auto m = a.size(0);
    auto k = a.size(1);
    auto n = b.size(1);

    auto c = torch::empty({m, n}, dtype, a.device());

    int block_dim = 32;
    dim3 grid(
        (n + block_dim - 1) / block_dim,
        (m + block_dim - 1) / block_dim,
        1
    );

    int lda = a.stride(0);
    int ldb = b.stride(0);
    int ldc = c.stride(0);

    if (dtype == torch::kFloat32) {
        fast_matmul_kernel<float><<<grid, dim3(block_dim, block_dim)>>>(
            a.data_ptr<float>(), 
            b.data_ptr<float>(), 
            c.data_ptr<float>(), 
            m, k, n, 
            lda, ldb, ldc
        );
    } else if (dtype == torch::kHalf) {
        fast_matmul_kernel<__half><<<grid, dim3(block_dim, block_dim)>>>(
            reinterpret_cast<const __half*>(a.data_ptr()),
            reinterpret_cast<const __half*>(b.data_ptr()),
            reinterpret_cast<__half*>(c.data_ptr()),
            m, k, n,
            lda, ldb, ldc
        );
    }

    return c;
}
"""

matrix_multiplication_cpp = """
torch::Tensor fast_matmul_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::ScalarType dtype = torch::kFloat32
);
"""

fast_matmul = load_inline(
    name="fast_matmul",
    cpp_sources=matrix_multiplication_cpp,
    cuda_sources=matrix_multiplication_source,
    functions=["fast_matmul_cuda"],
    verbose=True,
    extra_cflags=["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-gencode=arch=compute_80,code=sm_80"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.fast_matmul = fast_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.fast_matmul.fast_matmul_cuda(A.t(), B.t(), torch.float32).t()

def get_inputs():
    A = torch.rand(K, M).cuda()  # Note that the original code requires transpose, so inputs should be in the correct shape
    B = torch.rand(N, K).cuda()
    return [A, B]

def get_init_inputs():
    return []