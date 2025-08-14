import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename scalar_t>
__global__ void matmul_kernel(const scalar_t* __restrict__ A,
                             const scalar_t* __restrict__ B,
                             scalar_t* __restrict__ C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; k += 1) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

at::Tensor matmul_cuda(at::Tensor A, at::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    at::Tensor C = at::empty({M, N}, A.options());

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    // Launch the kernel with float type
    matmul_kernel<float><<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    return C;
}
"""

matmul_cpp_source = """
at::Tensor matmul_cuda(at::Tensor A, at::Tensor B);
"""

# Compile the inline CUDA code
matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_kernel,
    functions=["matmul_cuda"],
    verbose=False,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A.cuda(), B.cuda())

def get_inputs():
    M_val = 256
    N_val = 256
    K_val = 131072 * 4
    A = torch.rand(M_val, K_val).cuda()
    B = torch.rand(K_val, N_val).cuda()
    return [A, B]

def get_init_inputs():
    return []