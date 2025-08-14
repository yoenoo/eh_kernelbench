import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename T>
__global__ void matmul_kernel(const T* __restrict__ A, const T* __restrict__ B, T* C, 
                             int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; k++) {
            sum += static_cast<float>(A[row * K + k]) * 
                   static_cast<float>(B[k * N + col]);
        }
        C[row * N + col] = sum;
    }
}

at::Tensor matmul_cuda(at::Tensor A, at::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = at::empty({M, N}, A.options());

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    matmul_kernel<<<blocks, threads, 0, stream>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}
"""

matmul_cpp_source = "at::Tensor matmul_cuda(at::Tensor A, at::Tensor B);"

matmul_op = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=False,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A.cuda(), B.cuda())