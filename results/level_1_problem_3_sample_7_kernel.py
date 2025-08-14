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

__global__ void batched_matmul_kernel(
    const float* A, const float* B, float* C,
    int batch_size, int m, int k, int n
) {
    int batch_idx = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    if (batch_idx >= batch_size) return;

    __shared__ float shared_A[32][32];
    __shared__ float shared_B[32][32];

    float sum = 0.0;

    for (int tile = 0; tile < (k + 31) / 32; ++tile) {
        // Load tiles of A into shared memory
        if (row < m && (tile * 32 + threadIdx.x) < k) {
            shared_A[row][threadIdx.x] = A[
                batch_idx * m * k + row * k + tile * 32 + threadIdx.x
            ];
        } else {
            shared_A[row][threadIdx.x] = 0.0;
        }

        // Load tiles of B into shared memory
        if ((tile * 32 + threadIdx.y) < k && col < n) {
            shared_B[threadIdx.y][col] = B[
                batch_idx * k * n + (tile * 32 + threadIdx.y) * n + col
            ];
        } else {
            shared_B[threadIdx.y][col] = 0.0;
        }

        __syncthreads();

        // Compute the dot product for the current tile
        for (int i = 0; i < 32; ++i) {
            sum += shared_A[row][i] * shared_B[i][col];
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (row < m && col < n) {
        C[batch_idx * m * n + row * n + col] = sum;
    }
}

torch::Tensor batched_matmul_cuda(
    torch::Tensor A, torch::Tensor B,
    int batch_size, int m, int k, int n
) {
    dim3 threads(32, 32);  // 32x32 threads per block (for m x n output)
    dim3 blocks(batch_size);  // One block per batch

    auto C = torch::empty({batch_size, m, n}, A.options());

    batched_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        batch_size, m, k, n
    );

    return C;
}
"""

bmm_kernel_header = """
torch::Tensor batched_matmul_cuda(
    torch::Tensor A, torch::Tensor B,
    int batch_size, int m, int k, int n
);
"""

# Compile the CUDA kernel
batched_matmul = load_inline(
    name='batched_matmul',
    cpp_sources=bmm_kernel_header,
    cuda_sources=bmm_kernel_source,
    functions=['batched_matmul_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm = batched_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.bmm.batched_matmul_cuda(
            A, B, batch_size, m, k, n
        )

def get_inputs():
    A = torch.rand(batch_size, m, k).cuda()
    B = torch.rand(batch_size, k, n).cuda()
    return [A, B]

def get_init_inputs():
    return []