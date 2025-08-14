import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication (A * B.T)
matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matrix_mult_kernel(const scalar_t* __restrict__ a, const scalar_t* __restrict__ b, 
                                  scalar_t* __restrict__ c, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        scalar_t sum = 0.;
        for (int k = 0; k < K; k++) {
            sum += a[row * K + k] * b[col * K + k];
        }
        c[row * N + col] = sum;
    }
}

torch::Tensor matrix_mult_cuda(torch::Tensor a, torch::Tensor b, int M, int N, int K) {
    auto dtype = a.dtype();
    auto c = torch::empty({M, N}, dtype);

    int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((N + block_size - 1) / block_size, (M + block_size - 1) / block_size);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "matrix_mult_cuda", ([&] {
        matrix_mult_kernel<scalar_t><<<grid, block, 0, stream>>>(
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(), M, N, K);
    }));

    return c;
}
"""

matrix_mult_cpp_source = """
torch::Tensor matrix_mult_cuda(torch::Tensor a, torch::Tensor b, int M, int N, int K);
"""

# Compile the inline CUDA code for matrix multiplication
matrix_mult = load_inline(
    name="matrix_mult",
    cpp_sources=matrix_mult_cpp_source,
    cuda_sources=matrix_mult_source,
    functions=["matrix_mult_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix_mult = matrix_mult

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.size()
        N = B.size(0)
        return self.matrix_mult.matrix_mult_cuda(A, B, M, N, K)