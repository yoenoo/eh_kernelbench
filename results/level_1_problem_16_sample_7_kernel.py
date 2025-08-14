import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void fast_matmul_kernel(const scalar_t* __restrict__ a,
                                  const scalar_t* __restrict__ b,
                                  scalar_t* __restrict__ c,
                                  int m, int k, int n) {
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

torch::Tensor fast_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    const int m = a.size(0);
    const int k = a.size(1);
    const int n = b.size(1);

    auto c = torch::empty({m, n}, a.options());

    const int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((n + block_size - 1) / block_size, (m + block_size - 1) / block_size);

    AT_DISPATCH_ALL_TYPES(a.scalar_type(), "fast_matmul_cuda", ([&] {
        fast_matmul_kernel<scalar_t><<<grid, block>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            m, k, n);
    }));

    return c;
}
"""

matmul_kernel_cpp = """
torch::Tensor fast_matmul_cuda(torch::Tensor a, torch::Tensor b);
"""

fast_matmul = load_inline(
    name="fast_matmul",
    cpp_sources=matmul_kernel_cpp,
    cuda_sources=matmul_kernel_source,
    functions=["fast_matmul_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["--expt-relaxed-constexpr"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = fast_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Transpose A and perform custom matmul
        return self.matmul.fast_matmul_cuda(A.t(), B)