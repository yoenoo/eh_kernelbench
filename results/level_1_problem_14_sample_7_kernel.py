import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for upper triangular matrix multiplication
utmatmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void utmatmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        if (row <= col) {
            float sum = 0.0;
            for (int k = 0; k < N; k++) {
                int idx_A = row * N + k;
                int idx_B = k * N + col;
                if (k <= col) {  // Since B is upper triangular, B[k][col] is non-zero only if k <= col
                    sum += A[idx_A] * B[idx_B];
                }
            }
            C[row * N + col] = sum;
        }
    }
}

torch::Tensor utmatmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    const int threads = 32;
    dim3 blocks((N + threads - 1)/threads, (N + threads - 1)/threads);
    dim3 threadsPerBlock(threads, threads);

    auto C = torch::empty({N, N}, A.options());
    utmatmul_kernel<<<blocks, threadsPerBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    cudaDeviceSynchronize();
    return C;
}
"""

# Compile the custom CUDA kernel
utmatmul = load_inline(
    name="utmatmul",
    cpp_sources="",
    cuda_sources=utmatmul_source,
    functions=["utmatmul_cuda"],
    verbose=True,
    extra_cflags=["-D__CUDA_NO_HALF_OPERATORS__"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.utmatmul = utmatmul

    def forward(self, A, B):
        return self.utmatmul.utmatmul_cuda(A, B)

def get_inputs():
    N = 4096
    A = torch.triu(torch.rand(N, N).cuda())
    B = torch.triu(torch.rand(N, N).cuda())
    return [A, B]

def get_init_inputs():
    return []