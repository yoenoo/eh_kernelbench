import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication with optimized dimensions
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void optimized_matmul_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, 
                                        int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0;
        for (int e = 0; e < k; e++) {
            sum += a[row * k + e] * b[e * n + col];
        }
        c[row * n + col] = sum;
    }
}

torch::Tensor optimized_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    const int m = a.size(0);
    const int k = a.size(1);
    const int n = b.size(1);
    auto c = torch::empty({m, n}, torch::device("cuda"));

    dim3 threads(32, 8);
    dim3 blocks((n + threads.x - 1)/threads.x, (m + threads.y - 1)/threads.y);

    optimized_matmul_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), 
                                                 c.data_ptr<float>(), m, n, k);

    cudaDeviceSynchronize();
    return c;
}
"""

matmul_cpp_source = "torch::Tensor optimized_matmul_cuda(torch::Tensor a, torch::Tensor b);"

# Compile the inline CUDA code
optimized_matmul = load_inline(
    name="optimized_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["optimized_matmul_cuda"],
    verbose=True,
    extra_cflags=["-DENABLE什么东西什么优化"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = optimized_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.optimized_matmul_cuda(A.cuda(), B.cuda())

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []