import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

# Custom CUDA kernel for batched matrix multiplication (bmm)
batch_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batch_matmul_kernel(const float* A, const float* B, float* C,
                                    int batch_size, int m, int k, int n) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batch_size) return;

    for (int i = threadIdx.y; i < m; i += blockDim.y) {
        for (int j = threadIdx.z; j < n; j += blockDim.z) {
            float sum = 0.0;
            for (int l = 0; l < k; ++l) {
                sum += A[batch * m * k + i * k + l] * B[batch * k * n + l * n + j];
            }
            C[batch * m * n + i * n + j] = sum;
        }
    }
}

torch::Tensor batch_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int batch_size = A.size(0);
    const int m = A.size(1);
    const int k = A.size(2);
    const int n = B.size(2);

    auto C = torch::empty({batch_size, m, n}, A.options());

    dim3 threads(1, 32, 32); // Thread block dimensions (z,y,x) optimized for 3D grid
    dim3 blocks(batch_size, 1, 1); // One block per batch sample

    batch_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        batch_size, m, k, n
    );

    return C;
}
"""

batch_matmul_cpp_source = "torch::Tensor batch_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for batched matrix multiplication
batch_matmul = load_inline(
    name="batch_matmul",
    cpp_sources=batch_matmul_cpp_source,
    cuda_sources=batch_matmul_source,
    functions=["batch_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batch_matmul = batch_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batch_matmul.batch_matmul_cuda(A, B)

def get_inputs():
    A = torch.rand(batch_size, m, k).cuda()
    B = torch.rand(batch_size, k, n).cuda()
    return [A, B]

def get_init_inputs():
    return []