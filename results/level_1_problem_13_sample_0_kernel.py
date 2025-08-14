import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for symmetric matrix multiplication
symmetric_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Assuming row-major storage
template <typename scalar_t>
__global__ void symmetric_matmul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        scalar_t sum = 0;
        // Since both A and B are symmetric, exploit symmetry to reduce computation
        for (int k = 0; k < N; k += 1) {
            // Take advantage of symmetry in A and B
            sum += A[row * N + k] * B[col * N + k]; // C[row][col] = A row dot B col
            // Due to symmetry, C[col][row] can be calculated with same sum as above, but we will write it in another thread
        }
        C[row * N + col] = sum;
        // Write the symmetric counterpart here if beneficial (depends on memory access pattern)
        // However, this may cause divergence, so omitted for now
    }
}

at::Tensor symmetric_matmul_cuda(at::Tensor A, at::Tensor B) {
    const int N = A.size(0);
    const int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((N + block_size - 1)/block_size, (N + block_size - 1)/block_size);

    at::Tensor C = at::empty({N, N}, A.options());

    symmetric_matmul_kernel<<<grid, block>>>(
        A.data_ptr<scalar_t>(),
        B.data_ptr<scalar_t>(),
        C.data_ptr<scalar_t>(),
        N);

    cudaDeviceSynchronize();
    return C;
}

// Handler for float tensors
at::Tensor symmetric_matmul_cuda_float(at::Tensor A, at::Tensor B) {
    return symmetric_matmul_cuda<float>(A, B);
}

// Handler for half tensors (if needed)
at::Tensor symmetric_matmul_cuda_half(at::Tensor A, at::Tensor B) {
    return symmetric_matmul_cuda<at::Half>(A, B);
}
"""

# Declaration for both float and half versions
symmetric_matmul_decl = """
torch::Tensor symmetric_matmul_cuda_float(torch::Tensor A, torch::Tensor B);
torch::Tensor symmetric_matmul_cuda_half(torch::Tensor A, torch::Tensor B);
"""

# Compile the CUDA extension
symmetric_matmul = load_inline(
    name="symmetric_matmul",
    cpp_sources=symmetric_matmul_decl,
    cuda_sources=symmetric_matmul_source,
    functions=['symmetric_matmul_cuda_float', 'symmetric_matmul_cuda_half'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.symmetric_matmul = symmetric_matmul

    def forward(self, A, B):
        if A.dtype == torch.float32:
            return self.symmetric_matmul.symmetric_matmul_cuda_float(A.cuda(), B.cuda())
        elif A.dtype == torch.float16:
            return self.symmetric_matmul.symmetric_matmul_cuda_half(A.cuda(), B.cuda())
        else:
            raise NotImplementedError("Only float32 and float16 are supported")