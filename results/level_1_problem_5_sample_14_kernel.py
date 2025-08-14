import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for scalar multiplication
scalar_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scalar_mult_kernel(const float* a, float scalar, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * scalar;
    }
}

extern "C" {
    torch::Tensor scalar_mult_cuda(torch::Tensor a, float s) {
        auto size = a.numel();
        auto out = torch::empty_like(a);

        const int block_size = 256;
        int num_blocks = (size + block_size - 1) / block_size;

        scalar_mult_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), s, out.data_ptr<float>(), size);

        return out;
    }
}
"""

scalar_mult_cpp_source = "torch::Tensor scalar_mult_cuda(torch::Tensor a, float s);"

# Compile the inline CUDA code for scalar multiplication
scalar_mult = load_inline(
    name="scalar_mult",
    cpp_sources=scalar_mult_cpp_source,
    cuda_sources=scalar_mult_source,
    functions=["scalar_mult_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar_mult = scalar_mult

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.scalar_mult.scalar_mult_cuda(A, s)