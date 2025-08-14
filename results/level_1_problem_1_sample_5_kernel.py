import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_mult_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B, int N) {
    auto C = torch::empty(N, N, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    matrix_mult_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

matrix_mult_cpp_source = "torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B, int N);"

# Compile the inline CUDA code for matrix multiplication
matrix_mult = load_inline(
    name="matrix_mult",
    cpp_sources=matrix_mult_cpp_source,
    cuda_sources=matrix_mult_source,
    functions=["matrix_mult_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_mult = matrix_mult
        self.N = 2048 * 2  # Ensure N is known at kernel launch time

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrix_mult.matrix_mult_cuda(A.cuda(), B.cuda(), self.N)

def get_inputs():
    A = torch.rand(N, N).cuda()
    B = torch.rand(N, N).cuda()
    return [A, B]