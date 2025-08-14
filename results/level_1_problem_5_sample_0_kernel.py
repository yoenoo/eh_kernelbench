import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

scalar_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scalar_mult_kernel(const float* A, float s, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = A[idx] * s;
    }
}

torch::Tensor scalar_mult_cuda(torch::Tensor A, float s) {
    auto size = A.numel();
    auto out = torch::empty_like(A);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    scalar_mult_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), s, out.data_ptr<float>(), size);

    return out;
}
"""

scalar_mult_cpp_source = "torch::Tensor scalar_mult_cuda(torch::Tensor A, float s);"

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
        super(ModelNew, self).__init__()
        self.scalar_mult = scalar_mult

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.scalar_mult.scalar_mult_cuda(A, s)