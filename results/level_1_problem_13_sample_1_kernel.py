import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class CustomMatmulCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return matmul_cuda(A, B)

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = matmul_cuda(grad_output, B.t())
        grad_B = matmul_cuda(A.t(), grad_output)
        return grad_A, grad_B

# Define the custom CUDA kernel for matrix multiplication of symmetric matrices
matmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
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

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    const int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((N + block_size - 1)/block_size, (N + block_size - 1)/block_size);

    auto C = torch::empty({N, N}, A.options());
    
    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    cudaDeviceSynchronize();
    return C;
}
"""

# Compile the inline CUDA code for optimized matrix multiplication
matmul_cuda = load_inline(
    name='matmul_cuda',
    cpp_sources="",
    cuda_sources=matmul_source,
    functions=['matmul_cuda'],
    verbose=True,
    with_cuda=True
).matmul_cuda

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
    def forward(self, A, B):
        return CustomMatmulCUDA.apply(A.cuda(), B.cuda())