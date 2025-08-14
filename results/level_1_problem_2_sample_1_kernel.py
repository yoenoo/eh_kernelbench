cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

template<typename scalar_t>
__global__ void matmul_kernel(const scalar_t* __restrict__ A,
                             const scalar_t* __restrict__ B,
                             scalar_t* __restrict__ C,
                             int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    const int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);

    using scalar_t = float;
    matmul_kernel<<<grid, block, 0, c10::cuda::current_stream()>>>((scalar_t*)A.data_ptr(),
            (scalar_t*)B.data_ptr(), (scalar_t*)C.data_ptr(), M, K, N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda", &matmul_cuda, "CUDA matmul implementation");
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

matmul_extension = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_kernel_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-g"],
    extra_cuda_cflags=['--expt-extended-lambda']
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_op = matmul_extension

    def forward(self, A, B):
        return self.matmul_op.matmul_cuda(A, B)

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []