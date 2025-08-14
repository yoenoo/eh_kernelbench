import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardSigmoid
hardsigmoid_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void hardsigmoid_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = x[idx] + 3.0f;
        if (z < 0.0f) {
            y[idx] = 0.0f;
        } else if (z > 6.0f) {
            y[idx] = 1.0f;
        } else {
            y[idx] = z / 6.0f;
        }
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
hardsigmoid_cuda = load_inline(
    name="hardsigmoid_cuda",
    cpp_sources=hardsigmoid_cpp_source,
    cuda_sources=hardsigmoid_source,
    functions=["hardsigmoid_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardsigmoid_cuda = hardsigmoid_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardsigmoid_cuda.hardsigmoid_cuda(x)