import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_tall_skinny_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 256

template<typename scalar_t>
__global__ void matmul_kernel(scalar_t* __restrict__ A, scalar_t* __restrict__ B, scalar_t* __restrict__ C,
                            int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        scalar_t sum = 0.0;
        for (int e = 0; e < k; ++e) {
            sum += A[row * k + e] * B[e * n + col];
        }
        C[row * n + col] = sum;
    }
}

torch::Tensor matmul_tall_skinny_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(BLOCK_SIZE, 1);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    const int block_size = 256;
    const int num_blocks = (M*N + block_size - 1)/block_size;

    matmul_kernel<float><<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N
    );

    cudaDeviceSynchronize();
    return C;
}
"""

matmul_tall_skinny_cpp_source = """
torch::Tensor matmul_tall_skinny_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_tall_skinny = load_inline(
    name="matmul_tall_skinny",
    cpp_sources=matmul_tall_skinny_cpp_source,
    cuda_sources=matmul_tall_skinny_source,
    functions=["matmul_tall_skinny_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul_tall_skinny

    def forward(self, A, B):
        return self.matmul.matmul_tall_skinny_cuda(A, B)

def get_inputs():
    M = 16384 * 2
    N = 16 * 2
    A = torch.rand(M, N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]

def get_init_inputs():
    return []

# Update the get_inputs to move tensors to CUDA
original_get_inputs = get_inputs
def get_inputs():
    A, B = original_get_inputs()
    return [A.cuda(), B.cuda()]
get_inputs = get_inputs  # Rebind to new function