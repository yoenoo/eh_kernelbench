import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N)
        return;
    
    int i = idx / N;
    int j = idx % N;
    
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
    }
    C[idx] = sum;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure the tensors are contiguous
    A = A.contiguous();
    B = B.contiguous();
    
    // Get dimensions
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Incompatible dimensions for matrix multiplication");
    
    auto C = torch::empty({M, N}, A.options());
    
    int threads_per_block = 256;
    int num_elements = M * N;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    
    matmul_kernel<<<blocks_per_grid, threads_per_block>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(),
        M, K, N
    );
    
    // Check for errors in the kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }
    
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the CUDA extension
matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A, B)