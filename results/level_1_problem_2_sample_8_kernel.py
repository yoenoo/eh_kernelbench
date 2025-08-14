import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_matmul_kernel(const float* A, const float* B, float* C,
                                    int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor custom_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    auto C = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 threads(16, 16);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    custom_matmul_kernel<<<blocks, threads, 0, stream>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N
    );

    return C;
}
"""

matmul_kernel_header = """
torch::Tensor custom_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_op = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_kernel_header,
    cuda_sources=matmul_kernel_source,
    functions=["custom_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_op.custom_matmul_cuda(A.cuda(), B.cuda())