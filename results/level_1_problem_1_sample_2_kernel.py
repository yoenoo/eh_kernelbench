import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for matrix multiplication
matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C, const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int N) {
    auto C = torch::empty({N, N}, A.options());
    
    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    matmul_kernel<float><<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    cudaDeviceSynchronize();
    
    return C;
}
"""

matmul_kernel_cpp = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int N);"

matmul_cuda = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_kernel_cpp,
    cuda_sources=matmul_kernel_source,
    functions=["matmul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.N = 2048 * 2
        self.matmul_cuda = matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Assuming input tensors are already on the GPU
        A = A.cuda()
        B = B.cuda()
        return self.matmul_cuda.matmul_cuda(A, self.N)