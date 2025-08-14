import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for lower triangular matrix multiplication
lower_triangular_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

template<typename scalar_t>
__global__ void lower_triangular_matmul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < N; k++) {
            // Since A and B are lower triangular, A[row][k] is zero when k > row
            // and B[k][col] is zero when k > col. Thus, the product is non-zero only when k <= row and k <= col.
            // Additionally, the result C[row][col] is only lower triangular when row >= col.
            // Hence, we can skip computations where row < col.
            if (row >= col && k <= row && k <= col) {
                sum += A[row * N + k] * B[k * N + col];
            }
        }
        if (row >= col) {
            C[row * N + col] = sum;
        }
    }
}

template<typename scalar_t>
torch::Tensor lower_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    const int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((N + block_size - 1)/block_size, (N + block_size - 1)/block_size);

    auto C = torch::empty({N, N}, A.options());
    lower_triangular_matmul_kernel<scalar_t><<<grid, block>>>(
        A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(), N);
    return C;
}

torch::Tensor lower_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    auto device = A.device();
    if (A.device() != B.device()) {
        throw "Tensors must be on the same device!";
    }
    if (A.size(0) != A.size(1) || B.size(0) != B.size(1)) {
        throw "Input tensors must be square matrices!";
    }
    if (A.size(0) != B.size(0)) {
        throw "Input matrices must have the same size!";
    }

    if (A.scalar_type() == torch::kFloat32) {
        return lower_triangular_matmul_cuda<float>(A, B);
    } else {
        // Add support for other types if necessary
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "lower_triangular_matmul", ([&] {
            return lower_triangular_matmul_cuda<scalar_t>(A, B);
        }));
    }

    return torch::Tensor(); // Should never reach here
}
"""

lower_triangular_matmul_cpp_source = """
torch::Tensor lower_triangular_matmul(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
lower_triangular_matmul = load_inline(
    name="lower_triangular_matmul",
    cpp_sources=lower_triangular_matmul_cpp_source,
    cuda_sources=lower_triangular_matmul_source,
    functions=["lower_triangular_matmul"],
    verbose=False,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-lineinfo", "-arch=sm_70"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.ltm = lower_triangular_matmul

    def forward(self, A, B):
        return self.ltm(A.cuda(), B.cuda())