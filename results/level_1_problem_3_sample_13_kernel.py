import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

def get_batch_gemm_kernel():
    kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

typedef char byte;

#define CUDA_1D_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;     \
      i < (n);                                           \
      i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void batch_gemm_forward_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int batch_size, int m, int k, int n) {

  CUDA_1D_KERNEL_LOOP(index, batch_size * m * n) {
    int bid = index / (m * n);
    int midn = index % (m * n);
    int mid = midn / n;
    int nid = midn % n;

    scalar_t sum = 0;
    for (int kid = 0; kid < k; ++kid) {
      scalar_t a = A[bid * m * k + mid * k + kid];
      scalar_t b = B[bid * k * n + kid * n + nid];
      sum += a * b;
    }
    C[index] = sum;
  }
}

torch::Tensor batch_gemm_forward(torch::Tensor A, torch::Tensor B) {
    const auto batch_size = A.size(0);
    const auto m = A.size(1);
    const auto k = A.size(2);
    const auto n = B.size(2);

    auto C = torch::empty({batch_size, m, n}, A.options());

    const int threads = 256;
    const int elements = batch_size * m * n;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(A.type(), "batch_gemm_forward", ([&] {
        batch_gemm_forward_kernel<scalar_t><<<blocks, threads>>>(
            A.data<scalar_t>(), 
            B.data<scalar_t>(), 
            C.data<scalar_t>(),
            batch_size, m, k, n);
    }));

    cudaDeviceSynchronize();
    return C;
}
    """
    return kernel

batch_gemm_source = get_batch_gemm_kernel()
batch_gemm_header = (
    "torch::Tensor batch_gemm_forward(torch::Tensor A, torch::Tensor B);"
)

# Compile the CUDA kernel
batch_gemm = load_inline(
    name="batch_gemm",
    cpp_sources=batch_gemm_header,
    cuda_sources=batch_gemm_source,
    functions=["batch_gemm_forward"],
    verbose=True,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_gemm = batch_gemm

    def forward(self, A, B):
        return self.batch_gemm.batch_gemm_forward(A.cuda(), B.cuda()).cuda()