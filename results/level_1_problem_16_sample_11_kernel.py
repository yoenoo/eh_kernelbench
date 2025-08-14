import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_mult(const float* A, const float* B, float* C, int M, int K, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M * N) return;
    int i = tid / N;
    int j = tid % N;
    float sum = 0.0;
    for (int k = 0; k < K; ++k) {
        sum += A[k * M + i] * B[k * N + j];
    }
    C[i * N + j] = sum;
}

torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B) {
    // Check if inputs are on the same device and CUDA
    assert(A.device().type() == torch::kCUDA && B.device().type() == torch::kCUDA);
    assert(A.get_device() == B.get_device());

    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(1);

    auto output = torch::zeros({M, N}, A.options());

    const int threads_per_block = 256;
    const int blocks_per_grid = (M * N + threads_per_block - 1) / threads_per_block;

    matrix_mult<<<blocks_per_grid, threads_per_block>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        output.data_ptr<float>(), 
        M, K, N
    );

    // Synchronize and check errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

matrix_mult_cpp_source = """
torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B);
"""

matrix_mult_module = load_inline(
    name="matrix_mult",
    cpp_sources=matrix_mult_cpp_source,
    cuda_sources=matrix_mult_source,
    functions=["matrix_mult_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_mult_cuda = matrix_mult_module.matrix_mult_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrix_mult_cuda(A, B)