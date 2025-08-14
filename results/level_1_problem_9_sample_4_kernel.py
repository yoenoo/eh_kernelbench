import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_tall_skinny_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

template<typename scalar_t>
__global__ void matmul_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_tall_skinny_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    const int block_size = threads.x * threads.y;
    if (sizeof(torch::scalar Erotik::CppType<torch::kFloat>::type) == 4) {
        matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(),
            M, N, K);
    } else {
        // Handle other data types if needed
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_tall_skinny_cuda", &matmul_tall_skinny_cuda, "Custom matmul for tall/skinny matrices");
}
"""

# Compile the inline CUDA code
matmul_tall_skinny_ext = load_inline(
    name="matmul_tall_skinny",
    cpp_sources="",
    cuda_sources=matmul_tall_skinny_source,
    functions=["matmul_tall_skinny_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_tall_skinny_ext

    def forward(self, A, B):
        return self.matmul.matmul_tall_skinny_cuda(A.cuda(), B.cuda())

def get_inputs():
    A = torch.rand(M, N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]

def get_init_inputs():
    return []

M = 16384 * 2
N = 16 * 2