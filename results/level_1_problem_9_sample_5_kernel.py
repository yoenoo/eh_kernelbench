import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for tall and skinny matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void tall_skinny_matmul_kernel(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        T sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

torch::Tensor tall_skinny_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    const int m = a.size(0);
    const int k = a.size(1);
    const int n = b.size(1);

    auto c = torch::empty({m, n}, a.options());
    
    dim3 threads(32, 8);
    dim3 blocks((n + threads.x - 1)/threads.x, (m + threads.y - 1)/threads.y);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    if (a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32) {
        tall_skinny_matmul_kernel<float><<<blocks, threads, 0, stream>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, k, n);
    } else if (a.dtype() == torch::kHalf && b.dtype() == torch::kHalf) {
        tall_skinny_matmul_kernel<__half><<<blocks, threads, 0, stream>>>(reinterpret_cast<const __half*>(a.data_ptr()),
        reinterpret_cast<const __half*>(b.data_ptr()),
        reinterpret_cast<__half*>(c.data_ptr()),
        m, k, n);
    }

    return c;
}
"""

matmul_cpp_source = "torch::Tensor tall_skinny_matmul_cuda(torch::Tensor a, torch::Tensor b);"

# Compile the inline CUDA code
tall_skinny_matmul = load_inline(
    name="tall_skinny_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["tall_skinny_matmul_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA", "-x=cu"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = tall_skinny_matmul

    def forward(self, A, B):
        return self.matmul.tall_skinny_matmul_cuda(A.cuda(), B.cuda())

M = 16384 * 2
N = 16 * 2

def get_inputs():
    A = torch.rand(M, N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]

def get_init_inputs():
    return []