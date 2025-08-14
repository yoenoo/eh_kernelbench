import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* __restrict__ a, const scalar_t* __restrict__ b, scalar_t* __restrict__ c, const int M, const int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < 1) {
        scalar_t sum = 0;
        for (int k = 0; k < K; k++) {
            sum += a[row * K + k] * b[k];
        }
        c[row] = sum;
    }
}

template <typename scalar_t>
torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    const int M = a.size(0);
    const int K = a.size(1);
    torch::Tensor c = torch::empty({M, 1}, a.options());

    const int block_size_x = 32;
    const int block_size_y = 32;
    dim3 block(block_size_x, block_size_y);
    dim3 grid((1 + block_size_x - 1)/block_size_x, (M + block_size_y - 1)/block_size_y);

    matmul_kernel<scalar_t><<<grid, block>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(), M, K);
    cudaDeviceSynchronize();

    return c;
}

torch::Tensor matmul_forward(torch::Tensor a, torch::Tensor b) {
    auto a_contig = a.contiguous();
    auto b_contig = b.contiguous();

    if (a.dtype() == torch::kFloat32) {
        return matmul_cuda<float>(a_contig, b_contig);
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
}
"""

matmul_kernel_header = """
torch::Tensor matmul_forward(torch::Tensor a, torch::Tensor b);
"""

matmul_op = load_inline(
    name="matmul_op",
    cpp_sources=matmul_kernel_header,
    cuda_sources=matmul_kernel_source,
    functions="matmul_forward",
    verbose=True,
    extra_cuda_cflags=['-arch=sm_86'],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.cuda()
        B = B.cuda()
        return self.matmul_op.matmul_forward(A, B).view(-1, 1)