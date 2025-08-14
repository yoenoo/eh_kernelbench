import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fast_matmul_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C,
                                  int M, int N, int K) {
    __shared__ scalar_t shared_A[32][32];
    __shared__ scalar_t shared_B[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bdx = blockDim.x;
    int bdy = blockDim.y;

    int idx_x = blockIdx.x * bdx + tx;
    int idx_y = blockIdx.y * bdy + ty;

    scalar_t sum = 0.0;

    for (int m = 0; m < (K + bdx - 1) / bdx; ++m) {
        if (idx_x < M && m * bdx + ty < K) {
            shared_A[ty][tx] = A[idx_x * K + m * bdx + ty];
        } else {
            shared_A[ty][tx] = 0.0;
        }

        if (idx_y < N && m * bdx + tx < K) {
            shared_B[ty][tx] = B[(m * bdx + tx) * N + idx_y];
        } else {
            shared_B[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < bdx; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads();
    }

    if (idx_x < M && idx_y < N) {
        C[idx_x * N + idx_y] = sum;
    }
}

torch::Tensor fast_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto options = torch::device("cuda").dtype(A.dtype());
    torch::Tensor C = torch::zeros({M, N}, options);

    dim3 threads(32, 32);
    dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    fast_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

matmul_cpp_source = """
torch::Tensor fast_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
fast_matmul = load_inline(
    name="fast_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_kernel_source,
    functions=["fast_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.fast_matmul = fast_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.fast_matmul.fast_matmul_cuda(A.cuda(), B.cuda())

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []