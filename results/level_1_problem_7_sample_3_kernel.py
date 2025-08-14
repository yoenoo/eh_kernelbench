import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_matmul_kernel(const float* A, const float* B, float* C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    auto C = torch::empty({M, N}, torch::device("cuda"));

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    custom_matmul_kernel<<<blocks, threads, 0, stream>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

matmul_op = load_inline(
    name="matmul_op",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_kernel_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_op.matmul_cuda(A.cuda(), B.cuda())

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []

M = 16384 * 2
N = 16384 * 2
K = 32 * 2