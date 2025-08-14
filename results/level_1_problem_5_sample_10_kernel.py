import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_scalar_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_scalar_mult_kernel(const float* a, float s, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * s;
    }
}

torch::Tensor elementwise_scalar_mult_cuda(torch::Tensor a, float s) {
    auto size = a.numel();
    auto out = torch::empty_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_scalar_mult_kernel<<<num_blocks, block_size>>>(
        a.data_ptr<float>(), s, out.data_ptr<float>(), size
    );

    return out;
}
"""

elementwise_scalar_mult_cpp_source = (
    "torch::Tensor elementwise_scalar_mult_cuda(torch::Tensor a, float s);"
)

elementwise_scalar_mult = load_inline(
    name="elementwise_scalar_mult",
    cpp_sources=elementwise_scalar_mult_cpp_source,
    cuda_sources=elementwise_scalar_mult_source,
    functions=["elementwise_scalar_mult_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar_mult = elementwise_scalar_mult

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.scalar_mult.elementwise_scalar_mult_cuda(A, s)