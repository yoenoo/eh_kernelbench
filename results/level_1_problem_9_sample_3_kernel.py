import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for matrix multiplication optimized for tall and skinny matrices
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename T>
__global__ void matmul_tall_skinny_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
                                          int M, int K, int N, int lda, int ldb, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T val = 0;
        for (int k = 0; k < K; ++k) {
            val += A[row * lda + k] * B[k * ldb + col];
        }
        C[row * ldc + col] = val;
    }
}

template<typename T>
void matmul_tall_skinny_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    const int threads = 32;
    dim3 block(threads, threads);
    int grid_x = (N + threads - 1) / threads;
    int grid_y = (M + threads - 1) / threads;
    dim3 grid(grid_x, grid_y);

    const int lda = A.stride(0);
    const int ldb = B.stride(0);
    const int ldc = C.stride(0);

    if (std::is_same<T, half>::value) {
        matmul_tall_skinny_kernel<T><<<grid, block, 0, nullptr>>>(
            (const T*)A.data_ptr(), (const T*)B.data_ptr(), (T*)C.data_ptr(),
            M, K, N, lda, ldb, ldc);
    } else {
        matmul_tall_skinny_kernel<T><<<grid, block, 0, nullptr>>>(
            A.data_ptr<T>(), B.data_ptr<T>(), C.data_ptr<T>(),
            M, K, N, lda, ldb, ldc);
    }
}

torch::Tensor matmul_tall_skinny(torch::Tensor A, torch::Tensor B) {
    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::empty({A.size(0), B.size(1)}, options);

    if (A.dtype() == torch::kFloat32) {
        matmul_tall_skinny_cuda<float>(A, B, C);
    } else if (A.dtype() == torch::kHalf) {
        matmul_tall_skinny_cuda<at::Half>(A, B, C);
    } else {
        AT_ERROR("Unsupported data type");
    }

    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_tall_skinny(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code for matrix multiplication
matmul_op = load_inline(
    name="matmul_tall_skinny",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_tall_skinny"],
    verbose=True,
    extra_cflags=["-g", "-Wno-redundant-decls"],
    extra_cuda_cflags=["-g", "--expt-relaxed-constexpr"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_op

    def forward(self, A, B):
        return self.matmul.matmul_tall_skinny(A, B)