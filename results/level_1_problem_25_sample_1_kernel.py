import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void swish_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid_x = 1.0 / (1.0 + expf(-x[idx]));
        y[idx] = x[idx] * sigmoid_x;
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

swish_cpp_source = "torch::Tensor swish_cuda(torch::Tensor x);"

# Compile the inline CUDA code
swish = load_inline(
    name="swish",
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_source,
    functions=["swish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.swish_cuda = swish

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish_cuda.swish_cuda(x)