import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_small_K_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* __restrict__ a, const scalar_t* __restrict__ b, scalar_t* __restrict__ c, const int M, const int N, const int K) {
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> matmul_cuda(torch::Tensor a, torch::Tensor b) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
    torch::Tensor c = torch::zeros({M, N}, options);

    int block_size = 32;
    dim3 blocks(TORCH_DIVUP(N, block_size), TORCH_DIVUP(M, block_size));
    dim3 threads(block_size, block_size);

    if (a.dtype() == torch::kFloat32) {
        matmul_kernel<float><<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
    } else if (a.dtype() == torch::kHalf) {
        matmul_kernel<__half><<<blocks, threads>>>(reinterpret_cast<const __half*>(a.data_ptr()),
                                                  reinterpret_cast<const __half*>(b.data_ptr()),
                                                  reinterpret_cast<__half*>(c.data_ptr()),
                                                  M, N, K);
    } else {
        throw std::runtime_error("Unsupported data type");
    }

    return std::make_tuple(c, a, b); // Return dummy inputs to prevent early free
}

// Dummy function for compilation
torch::Tensor dummy_func(torch::Tensor a, torch::Tensor b) {
    return a + b;
}
"""

matmul_small_K_header = """
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> matmul_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor dummy_func(torch::Tensor a, torch::Tensor b);
"""

matmul_extension = load_inline(
    name="matmul_extension",
    cpp_sources=matmul_small_K_header,
    cuda_sources=matmul_small_K_source,
    functions=["matmul_cuda", "dummy_func"],
    verbose=True,
    extra_cuda_cflags=['-gencode=arch=compute_80,code=sm_80', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_INSTRUCTIONS__'],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_extension.matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Execute the custom kernel and return only the result tensor
        return self.matmul(A, B)[0]