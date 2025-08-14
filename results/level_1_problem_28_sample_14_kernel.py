import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardSigmoid
hardsigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void hardsigmoid_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx] + 3.0f;
        val = fmaxf(0.0f, val);
        val = fminf(val / 6.0f, 1.0f);
        out[idx] = val;
    }
}

torch::Tensor hardsigmoid_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardsigmoid_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

cpp_source = "torch::Tensor hardsigmoid_cuda(torch::Tensor x);"

# Compile the inline CUDA code for HardSigmoid
hardsigmoid = load_inline(
    name="hardsigmoid",
    cpp_sources=cpp_source,
    cuda_sources=hardsigmoid_source,
    functions=["hardsigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardsigmoid = hardsigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardsigmoid.hardsigmoid_cuda(x)