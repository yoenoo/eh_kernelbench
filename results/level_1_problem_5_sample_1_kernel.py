import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for scalar multiplication
scalar_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scalar_mul_kernel(const float* a, float scalar, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * scalar;
    }
}

torch::Tensor scalar_mul_cuda(torch::Tensor a, float scalar) {
    auto size = a.numel();
    auto out = torch::empty({a.sizes()}, a.options());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    scalar_mul_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), scalar, out.data_ptr<float>(), size);

    return out;
}
"""

scalar_mul_cpp_source = (
    "torch::Tensor scalar_mul_cuda(torch::Tensor a, float scalar);"
)

# Compile the inline CUDA code for scalar multiplication
scalar_mul = load_inline(
    name="scalar_mul",
    cpp_sources=scalar_mul_cpp_source,
    cuda_sources=scalar_mul_source,
    functions=["scalar_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar_mul = scalar_mul

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.scalar_mul.scalar_mul_cuda(A, s)