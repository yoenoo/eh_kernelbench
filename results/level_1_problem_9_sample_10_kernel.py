import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication optimized for tall and skinny matrices
matmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_kernel(scalar_t* __restrict__ A, scalar_t* __restrict__ B, scalar_t* __restrict__ C,
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

std::vector<int64_t> get_launch_dims(int M, int N, int block_size) {
    dim3 block(block_size, block_size);
    dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);
    return {grid.x, grid.y, block.x, block.y};
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    int block_size = 32; // Tune this parameter for optimal performance based on matrix dimensions

    auto launch_dims = get_launch_dims(M, N, block_size);
    dim3 grid(launch_dims[0], launch_dims[1]);
    dim3 block(launch_dims[2], launch_dims[3]);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_cuda", ([&] {
        matmul_kernel<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    return C;
}
"""

matmul_cpp_source = """
std::vector<int64_t> get_launch_dims(int M, int N, int block_size);
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda = load_inline(
    name="matmul_opt",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-g", "-Wno-deprecated-gpu-template"],
    extra_cuda_cflags=["-std=c++14", "--expt-extended-lambda"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_cuda

    def forward(self, A, B):
        return self.matmul.matmul_cuda(A.cuda(), B.cuda())

# Maintaining the same input generation for consistency
def get_inputs():
    A = torch.rand(M, N)
    B = torch.rand(N, M)
    return [A.cuda(), B.cuda()]

def get_init_inputs():
    return []