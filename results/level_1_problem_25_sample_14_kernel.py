import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* x, float* y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xi = x[idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-xi));
        y[idx] = xi * sigmoid_val;
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto n = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}
"""

swish_cpp_source = "torch::Tensor swish_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Swish
swish = load_inline(
    name="swish",
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_source,
    functions=["swish_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.swish = swish

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish.swish_cuda(x)