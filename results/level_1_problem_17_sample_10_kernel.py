import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* __restrict__ a,
                             const scalar_t* __restrict__ b,
                             scalar_t* __restrict__ c,
                             int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        scalar_t sum = 0;
        for (int e = 0; e < k; ++e) {
            sum += a[row * k + e] * b[col * k + e];
        }
        c[row * n + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    const int m = a.size(0);
    const int k = a.size(1);
    const int n = b.size(0);

    auto c = torch::empty({m, n}, a.options());

    const int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((n + block.x - 1)/block.x, (m + block.y - 1)/block.y);

    matmul_kernel<float><<<grid, block>>>(a.data_ptr<float>(),
                                         b.data_ptr<float>(),
                                         c.data_ptr<float>(),
                                         m, n, k);

    return c;
}
"""

matmul_kernel_header = """
torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);
"""

matmul_op = load_inline(
    name="matmul_op",
    cpp_sources=matmul_kernel_header,
    cuda_sources=matmul_kernel_source,
    functions=["matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A, B.cuda().T)