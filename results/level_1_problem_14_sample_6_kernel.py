import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

upper_triangular_mm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void upper_triangular_matmul_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int n) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) return;

    if (row > col) {
        c[row * n + col] = 0.0;
        return;
    }

    scalar_t sum = 0.0;
    for (int k = row; k <= col; ++k) {  // Only compute relevant elements
        // Since A is upper triangular, a[row, k] is zero when k < row.
        // B is upper triangular, so b[k][col] is zero when k > col.
        sum += a[row * n + k] * b[k * n + col];
    }

    c[row * n + col] = sum;
}

torch::Tensor upper_triangular_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    const int n = a.size(0);
    const int block_size = 32;
    dim3 threads(block_size, block_size);
    dim3 blocks((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

    torch::Tensor c = torch::zeros({n, n}, a.options());

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "upper_triangular_matmul_cuda", ([&] {
        upper_triangular_matmul_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            n);
    }));

    return c;
}
"""

upper_triangular_mm_cpp_source = "torch::Tensor upper_triangular_matmul_cuda(torch::Tensor a, torch::Tensor b);"

upper_triangular_mm = load_inline(
    name="upper_triangular_mm",
    cpp_sources=upper_triangular_mm_cpp_source,
    cuda_sources=upper_triangular_mm_source,
    functions=["upper_triangular_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.upper_triangular_mm = upper_triangular_mm

    def forward(self, A, B):
        return self.upper_triangular_mm.upper_triangular_matmul_cuda(A, B)