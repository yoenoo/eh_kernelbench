import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>

template <typename scalar_t>
__global__ void fast_matmul_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor fast_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    const int TPB = 32;
    dim3 threads(TPB, TPB);
    dim3 blocks((N + TPB - 1) / TPB, (M + TPB - 1) / TPB);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (A.scalar_type() == torch::kFloat32) {
        fast_matmul_kernel<float><<<blocks, threads, 0, stream>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    } else if (A.scalar_type() == torch::kHalf) {  // Optional support for FP16
        fast_matmul_kernel<__half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(A.data_ptr<__half>()),
            reinterpret_cast<const __half*>(B.data_ptr<__half>()),
            reinterpret_cast<__half*>(C.data_ptr<__half>()), M, N, K);
    }

    return C;
}
"""

matmul_cpp_source = """
torch::Tensor fast_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

fast_matmul = load_inline(
    name="fast_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["fast_matmul_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_cuda_cflags=["-gencode=arch=compute_75,code=sm_75", "-lineinfo"],
    extra_ldflags=[],
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = fast_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.fast_matmul_cuda(A.cuda(), B.cuda())

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(N, K).cuda()
    return [A, B]

def get_init_inputs():
    return []