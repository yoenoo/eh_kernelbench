import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_irregular_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template <typename scalar_t>
__global__ void irregular_matmul_kernel(const scalar_t* __restrict__ A,
                                       const scalar_t* __restrict__ B,
                                       scalar_t* __restrict__ C,
                                       const int M, const int K, const int N) {

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_irregular_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    const int block_size_x = 32;
    const int block_size_y = 8;
    dim3 block(block_size_x, block_size_y);
    dim3 grid((N + block_size_x - 1) / block_size_x, (M + block_size_y - 1) / block_size_y);

    AT_DISPATCH_ALL_TYPES(A.scalar_type(), "matmul_irregular_cuda", ([&] {
        irregular_matmul_kernel<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    return C;
}
"""

matmul_irregular_header = """
torch::Tensor matmul_irregular_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_irregular = load_inline(
    name="matmul_irregular",
    cpp_sources=matmul_irregular_header,
    cuda_sources=matmul_irregular_source,
    functions=["matmul_irregular_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_cuda_cflags=["--expt-extended-lambda"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_irregular = matmul_irregular

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_irregular.matmul_irregular_cuda(A, B.cuda(), B.cuda())

def get_inputs():
    M = 8205
    K = 2949
    N = 5921
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []