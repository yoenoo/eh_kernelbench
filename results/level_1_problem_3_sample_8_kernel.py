import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

# Custom CUDA kernel for batched matrix multiplication
bmm_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T>
__global__ void batched_matmul_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
                                     int batch_size, int m, int k, int n) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        T sum = 0;
        for (int e = 0; e < k; ++e) {
            sum += A[batch * m * k + row * k + e] * B[batch * k * n + e * n + col];
        }
        C[batch * m * n + row * n + col] = sum;
    }
}

at::Tensor bmm_cuda(const at::Tensor A, const at::Tensor B) {
    const int batch_size = A.size(0);
    const int m = A.size(1);
    const int k = A.size(2);
    const int n = B.size(2);

    auto C = at::empty({batch_size, m, n}, A.options());

    dim3 threads(32, 8);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y, batch_size);

    AT_DISPATCH_ALL_TYPES(A.scalar_type(), "batched_matmul", ([&] {
        batched_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data<scalar_t>(), B.data<scalar_t>(), C.data<scalar_t>(),
            batch_size, m, k, n);
    }));

    cudaDeviceSynchronize();
    return C;
}
"""

bmm_kernel_cpp = """
at::Tensor bmm_cuda(const at::Tensor A, const at::Tensor B);
"""

bmm_module = load_inline(
    name="bmm_kernel",
    cpp_sources=bmm_kernel_cpp,
    cuda_sources=bmm_kernel_source,
    functions=["bmm_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm = bmm_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.bmm.bmm_cuda(A, B)