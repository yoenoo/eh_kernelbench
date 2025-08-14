import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-vector multiplication
matvecmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>

#define BLOCK_SIZE 256

template <typename scalar_t>
__global__ void matvecmul_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* C, int M, int K) {
    extern __shared__ scalar_t shared_memory[];
    scalar_t* temp = shared_memory;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    int row = blockIdx.x * blockDim.z + tz;
    int col = 0; // since B is a vector (K,1)

    scalar_t sum = 0.0;

    for (int k_block = 0; k_block < K; k_block += blockDim.x) {
        scalar_t a = A[row * K + k_block + tx];
        scalar_t b = B[(k_block + tx) * 1]; // B is Kx1

        __syncthreads();
        sum += a * b;
        __syncthreads();
    }

    C[row] = sum;
}

torch::Tensor matvecmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    
    torch::Tensor C = torch::empty({M, 1}, A.options());

    dim3 threadsPerBlock(256, 1, 1);
    dim3 numBlocks((M + threadsPerBlock.z - 1) / threadsPerBlock.z, 1, 1);

    matvecmul_kernel<float><<<numBlocks, threadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K);

    return C;
}
"""

matvecmul_cpp_source = (
    "torch::Tensor matvecmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
matvecmul = load_inline(
    name="matvecmul",
    cpp_sources=matvecmul_cpp_source,
    cuda_sources=matvecmul_source,
    functions=["matvecmul_cuda"],
    verbose=True,
    extra_cflags=["-arch=sm_86"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.cuda_matvecmul = matvecmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.cuda_matvecmul.matvecmul_cuda(A, B)