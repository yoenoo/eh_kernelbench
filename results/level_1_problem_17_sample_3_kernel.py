import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_custom(torch::Tensor A, torch::Tensor B, int M, int N, int K) {
    const int threads = 32;
    dim3 block(threads, threads);
    int grid_x = (N + threads - 1) / threads;
    int grid_y = (M + threads - 1) / threads;
    dim3 grid(grid_x, grid_y);

    auto C = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    cudaDeviceSynchronize();

    return C;
}
"""

matmul_kernel_cpp_source = "torch::Tensor matmul_custom(torch::Tensor A, torch::Tensor B, int M, int N, int K);"

matmul_kernel = load_inline(
    name="matmul_custom",
    cpp_sources=matmul_kernel_cpp_source,
    cuda_sources=matmul_kernel_source,
    functions=["matmul_custom"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_custom = matmul_kernel

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M = A.size(0)
        K = A.size(1)
        N = B.size(0)
        return self.matmul_custom.matmul_custom(A.cuda(), B.cuda(), M, N, K)

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(N, K).cuda()
    return [A, B]

def get_init_inputs():
    return []