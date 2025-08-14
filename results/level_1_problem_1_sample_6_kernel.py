import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void matrix_mult_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y);

    auto C = torch::empty({N, N}, torch::CUDA(A.device().index()));

    matrix_mult_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    cudaDeviceSynchronize();

    return C;
}
"""

matrix_mult_cpp_source = "torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B, int N);"

matrix_mult = load_inline(
    name="matrix_mult",
    cpp_sources=matrix_mult_cpp_source,
    cuda_sources=matrix_mult_source,
    functions=["matrix_mult_cuda"],
    verbose=False,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-gencode=arch=compute_80,code=sm_80"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_mult = matrix_mult
        global N
        N = 2048 * 2

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrix_mult.matrix_mult_cuda(A.cuda(), B.cuda(), N)