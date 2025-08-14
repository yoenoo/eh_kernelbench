import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for upper triangular matrix multiplication
utmatmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void utmatmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, const int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        if (row <= col) {
            float sum = 0.0;
            for (int k = 0; k <= col; ++k) {
                if (row <= k) {
                    sum += A[row * N + k] * B[k * N + col];
                }
            }
            C[row * N + col] = sum;
        }
    }
}

torch::Tensor utmatmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y);

    utmatmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    cudaDeviceSynchronize();

    return C;
}
"""

utmatmul_cpp_source = "torch::Tensor utmatmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for upper triangular matrix multiplication
utmatmul = load_inline(
    name="utmatmul",
    cpp_sources=utmatmul_cpp_source,
    cuda_sources=utmatmul_source,
    functions=["utmatmul_cuda"],
    verbose=True,
    extra_cflags=[""],
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
    A = torch.triu(torch.rand(N, N)).cuda()
    B = torch.triu(torch.rand(N, N)).cuda()
    return [A, B]

def get_init_inputs():
    return []