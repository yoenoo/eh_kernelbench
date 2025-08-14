import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D tensor-matrix multiplication
tensor_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void tensor_matmul_kernel(const float* A, const float* B, float* C,
                                    int N, int M, int K, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * M) return;

    int n = idx / M;
    int m = idx % M;
    for (int l = 0; l < L; l++) {
        float sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[n * M * K + m * K + k] * B[k * L + l];
        }
        C[idx * L + l] = sum;
    }
}

torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);
    
    auto C = torch::empty({N, M, L}, A.options());
    
    const int block_size = 256;
    const int num_elements = N * M;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    tensor_matmul_kernel<<<num_blocks, block_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        N, M, K, L
    );

    return C;
}
"""

tensor_matmul_cpp_source = "torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
tensor_matmul = load_inline(
    name="tensor_matmul",
    cpp_sources=tensor_matmul_cpp_source,
    cuda_sources=tensor_matmul_source,
    functions=["tensor_matmul_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-gencode=arch=compute_80,code=sm_80"]  # Adjust based on GPU
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tensor_matmul = tensor_matmul

    def forward(self, A, B):
        return self.tensor_matmul.tensor_matmul_cuda(A, B)