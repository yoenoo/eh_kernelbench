import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* __restrict__ a,
                             const scalar_t* __restrict__ b,
                             scalar_t* __restrict__ c,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

at::Tensor matmul_cuda(at::Tensor a, at::Tensor b) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    auto c = at::empty({M, N}, a.options());

    const int threads = 32;
    dim3 blocks(TORCH_EXT_NS::div_up(N, threads), TORCH_EXT_NS::div_up(M, threads));
    dim3 threadsPerBlock(threads, threads);

    AT_DISPATCH_ALL_TYPES(a.scalar_type(), "matmul_cuda", ([&] {
        matmul_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            M, N, K);
    }));

    return c;
}
"""

matmul_cpp_source = """
at::Tensor matmul_cuda(at::Tensor a, at::Tensor b);
"""

# Compile the inline CUDA code for matrix multiplication
matmul_extension = load_inline(
    name="matmul_ext",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_kernel_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-DDEBUG"],
    extra_cuda_cflags=["-lineinfo", "--use_fast_math"],
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_extension

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A, B)

def get_inputs():
    N = 2048 * 2
    A = torch.rand(N, N).cuda()
    B = torch.rand(N, N).cuda()
    return [A, B]

def get_init_inputs():
    return []