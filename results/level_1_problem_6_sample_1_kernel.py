import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* __restrict__ a,
                             const scalar_t* __restrict__ b,
                             scalar_t* __restrict__ c,
                             const int m, const int k, const int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        scalar_t sum = 0;
        for (int e = 0; e < k; ++e) {
            sum += a[row * k + e] * b[e * n + col];
        }
        c[row * n + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    auto c = torch::empty({M, N}, a.options());

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1)/threads.x, (M + threads.y - 1)/threads.y);

    if (a.dtype() == torch::kFloat32) {
        matmul_kernel<float><<<blocks, threads>>>(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, K, N);
    } else if (a.dtype() == torch::kHalf) {  // Uncomment if using half-precision
        matmul_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<const __half*>(a.data_ptr()),
            reinterpret_cast<const __half*>(b.data_ptr()),
            reinterpret_cast<__half*>(c.data_ptr()), M, K, N);
    }

    cudaDeviceSynchronize();
    return c;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);
"""

matmul_op = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A, B)