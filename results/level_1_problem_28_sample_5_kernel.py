import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardSigmoid
hardsigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardsigmoid_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xi = x[idx];
        y[idx] = fmaxf(0.0f, fminf(1.0f, xi * 0.2f + 0.5f));
    }
}

torch::Tensor hardsigmoid_cuda(torch::Tensor x) {
    auto n = x.numel();
    auto y = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    hardsigmoid_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

hardsigmoid_cpp_source = (
    "torch::Tensor hardsigmoid_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code
hardsigmoid = load_inline(
    name="hardsigmoid",
    cpp_sources=hardsigmoid_cpp_source,
    cuda_sources=hardsigmoid_source,
    functions=["hardsigmoid_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardsigmoid = hardsigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardsigmoid.hardsigmoid_cuda(x)