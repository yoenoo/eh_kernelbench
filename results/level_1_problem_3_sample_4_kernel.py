import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for batched matrix multiplication (bmm)
batched_matmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define THREADS 256

template <typename scalar_t>
__global__ void batched_matmul_kernel(const scalar_t* __restrict__ A,
                                     const scalar_t* __restrict__ B,
                                     scalar_t* __restrict__ C,
                                     const int batch_size,
                                     const int m,
                                     const int k,
                                     const int n) {
    int batch = blockIdx.z;
    int tid = threadIdx.x;
    __shared__ scalar_t shared_A[THREADS];
    __shared__ scalar_t shared_B[THREADS];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    scalar_t sum = 0.0;

    for (int i = 0; i < k; i += blockDim.x) {
        if (row < m && i + tid < k) {
            shared_A[threadIdx.x] = A[batch * m * k + row * k + i + tid];
        } else {
            shared_A[threadIdx.x] = 0.0;
        }

        if (col < n && i + tid < k) {
            shared_B[threadIdx.x] = B[batch * k * n + (i + tid) * n + col];
        } else {
            shared_B[threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int s = 0; s < blockDim.x; ++s) {
            if (row < m && col < n) {
                sum += shared_A[s] * shared_B[s];
            }
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[batch * m * n + row * n + col] = sum;
    }
}

torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int batch_size = A.size(0);
    const int m = A.size(1);
    const int k = A.size(2);
    const int n = B.size(2);

    auto C = torch::empty({batch_size, m, n}, A.options());

    dim3 threads(32, 8);
    dim3 blocks(
        (n + threads.x - 1) / threads.x,
        (m + threads.y - 1) / threads.y,
        batch_size
    );

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "batched_matmul_cuda", ([&] {
        batched_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            batch_size, m, k, n
        );
    }));

    return C;
}
"""

batched_matmul_cpp_source = "torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
batched_matmul = load_inline(
    name="batched_matmul",
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_source,
    functions=["batched_matmul_cuda"],
    verbose=True,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_matmul = batched_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batched_matmul.batched_matmul_cuda(A, B)