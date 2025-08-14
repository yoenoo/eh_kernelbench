import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void custom_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor custom_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1)/threads.x, (M + threads.y - 1)/threads.y);

    custom_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    cudaDeviceSynchronize();

    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor custom_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
custom_matmul = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["custom_matmul_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_matmul = custom_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.custom_matmul.custom_matmul_cuda(A.cuda(), B.cuda()).cpu()

# The get_inputs and get_init_inputs functions are not included in the model code but should remain as per original
def get_inputs():
    M = 16384 * 2
    N = 16384 * 2
    K = 32 * 2
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []