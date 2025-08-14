import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void fast_matmul_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M,
    int K,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += static_cast<double>(A[row * K + k]) * static_cast<double>(B[k * N + col]);
        }
        C[row * N + col] = static_cast<T>(sum);
    }
}

torch::Tensor fast_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B,
    bool use_half = false
) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto options = A.options();
    torch::Tensor C = torch::empty({M, N}, options);

    dim3 threads(32, 8);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (use_half) {
        fast_matmul_kernel<half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const half*>(A.data_ptr()),
            reinterpret_cast<const half*>(B.data_ptr()),
            reinterpret_cast<half*>(C.data_ptr()),
            M, K, N
        );
    } else {
        fast_matmul_kernel<float><<<blocks, threads, 0, stream>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, K, N
        );
    }

    return C;
}
"""

matmul_cpp_source = R"""
torch::Tensor fast_matmul_cuda(torch::Tensor A, torch::Tensor B, bool use_half = false);
"""

# Compile both kernels
fast_matmul = load_inline(
    name="fast_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_kernel_source,
    functions=["fast_matmul_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.fast_matmul = fast_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Determine if input tensors are half precision
        use_half = A.dtype == torch.float16

        # Move tensors to CUDA if not already
        A = A.cuda()
        B = B.cuda()

        return self.fast_matmul.fast_matmul_cuda(A, B, use_half)