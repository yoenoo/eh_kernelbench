import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU
relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = max(0.0f, x[idx]);
    }
}

torch::Tensor relu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    relu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

# C++ function declaration
relu_cpp_source = "torch::Tensor relu_cuda(torch::Tensor x);"

# Compile the inline CUDA code
elementwise_relu = load_inline(
    name="relu_cuda",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu_cuda = elementwise_relu  # The compiled module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_cuda.relu_cuda(x)