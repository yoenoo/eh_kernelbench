import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#define BLOCK_M 128
#define BLOCK_N 256
#define BLOCK_K 32

template <typename T>
__global__ void custom_matmul_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int m, int k, int n
) {
    __shared__ T shared_a[BLOCK_K * BLOCK_M];
    __shared__ T shared_b[BLOCK_K * BLOCK_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gid_x = blockIdx.x * BLOCK_M + ty;
    int gid_y = blockIdx.y * BLOCK_N + tx;

    T val = 0;

    for (int p = 0; p < (k + BLOCK_K - 1)/BLOCK_K; ++p) {
        if (ty < BLOCK_M && p*BLOCK_K + tx < k) {
            shared_a[ty * BLOCK_K + tx] = a[gid_x * k + p*BLOCK_K + tx];
        } else {
            shared_a[ty * BLOCK_K + tx] = 0;
        }

        if (tx < BLOCK_N && p*BLOCK_K + ty < k) {
            shared_b[ty * BLOCK_N + tx] = b[(p*BLOCK_K + ty) * n + gid_y];
        } else {
            shared_b[ty * BLOCK_N + tx] = 0;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_K; ++i) {
            val += shared_a[ty * BLOCK_K + i] * shared_b[i * BLOCK_N + tx];
        }
        __syncthreads();
    }

    if (gid_x < m && gid_y < n) {
        c[gid_x * n + gid_y] = val;
    }
}

torch::Tensor custom_matmul_cuda(
    torch::Tensor a,
    torch::Tensor b
) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    auto c = torch::empty({M, N}, a.options());

    dim3 threads(BLOCK_N, BLOCK_M);
    dim3 blocks(
        (M + BLOCK_M - 1)/BLOCK_M,
        (N + BLOCK_N - 1)/BLOCK_N
    );

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    custom_matmul_kernel<float><<<blocks, threads, 0, stream>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        M, K, N
    );

    return c;
}
"""

matmul_kernel_header = """
torch::Tensor custom_matmul_cuda(torch::Tensor a, torch::Tensor b);
"""

# Compile the custom kernel
custom_matmul = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_kernel_header,
    cuda_sources=matmul_kernel_source,
    functions=["custom_matmul_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = custom_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_op.custom_matmul_cuda(A.t(), B)