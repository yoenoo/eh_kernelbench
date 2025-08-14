import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void matmul_3d_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C,
                                int N, int M, int K, int L) {
    int n = blockIdx.z;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || l >= L) return;

    scalar_t sum = 0;
    for (int k = 0; k < K; k += 32) {
        __syncthreads();
        scalar_t a = A[n * M * K + m * K + k];
        scalar_t b = B[k * L + l];
        sum += a * b;
    }
    C[n * M * L + m * L + l] = sum;
}

torch::Tensor matmul_3d_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    auto C = torch::empty({N, M, L}, A.options());

    const int block_size_x = 32;
    const int block_size_y = 8;
    dim3 block(block_size_x, block_size_y);
    dim3 grid(L / block_size_x + 1, M / block_size_y + 1, N);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_3d_cuda", ([&] {
        matmul_3d_kernel<scalar_t><<<grid, block>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(), N, M, K, L);
    }));

    return C;
}
"""

matmul_3d_cpp_source = (
    "torch::Tensor matmul_3d_cuda(torch::Tensor A, torch::Tensor B);"
)

matmul_3d = load_inline(
    name="matmul_3d",
    cpp_sources=matmul_3d_cpp_source,
    cuda_sources=matmul_3d_source,
    functions=["matmul_3d_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_3d = matmul_3d

    def forward(self, A, B):
        return self.matmul_3d.matmul_3d_cuda(A, B)