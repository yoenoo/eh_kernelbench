import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardSigmoid
hardsigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardsigmoid_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = x[idx] * 0.1666666667f + 0.5f;  // α = 1/6 ≈ 0.1666666667, β = 0.5
        z = z > 1.0f ? 1.0f : z;
        y[idx] = z < 0.0f ? 0.0f : z;
    }
}

torch::Tensor hardsigmoid_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardsigmoid_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

hardsigmoid_cpp_source = "torch::Tensor hardsigmoid_cuda(torch::Tensor x);"

# Compile the inline CUDA code
hardsigmoid = load_inline(
    name="hardsigmoid",
    cpp_sources=hardsigmoid_cpp_source,
    cuda_sources=hardsigmoid_source,
    functions=["hardsigmoid_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardsigmoid = hardsigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardsigmoid.hardsigmoid_cuda(x)