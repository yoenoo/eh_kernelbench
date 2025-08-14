import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

diag_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void diag_matmul_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* C, int N, int M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    for (int col = 0; col < M; col++) {
        C[row * M + col] = A[row] * B[row * M + col];
    }
}

template <typename scalar_t>
torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto M = B.size(1);
    auto C = torch::empty({N, M}, A.options());

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    diag_matmul_kernel<scalar_t><<<blocks, threads>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(), N, M);

    cudaDeviceSynchronize();
    return C;
}

torch::Tensor diag_matmul(torch::Tensor A, torch::Tensor B) {
    auto device = A.device();
    auto options = torch::TensorOptions().dtype(A.dtype()).device(device);
    if (A.device().type() == torch::kCUDA) {
        if (A.dtype() == torch::kFloat32) {
            return diag_matmul_cuda<float>(A, B);
        } else if (A.dtype() == torch::kHalf) {
            return diag_matmul_cuda<half>(A, B);
        }
    } else {
        // CPU implementation if needed
        return torch::diag(A) * B;
    }
}
"""

diag_matmul_header = "torch::Tensor diag_matmul(torch::Tensor A, torch::Tensor B);"

diag_matmul_ops = load_inline(
    name="diag_matmul",
    cpp_sources=diag_matmul_header,
    cuda_sources=diag_matmul_source,
    functions=["diag_matmul"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.diag_matmul = diag_matmul_ops

    def forward(self, A, B):
        return self.diag_matmul.diag_matmul(A.cuda(), B.cuda())