cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for SELU activation
selu_kernel_source = """
#include <torch/extension.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

// SELU constants
const float alpha = 1.6732632423543772848170429916717f;
const float scale = 1.0507009873554804934193349852946f;

__global__ void selu_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xi = x[idx];
        y[idx] = xi > 0.f ? scale * xi : scale * (alpha * exp(xi) - alpha);
    }
}

torch::Tensor selu_cuda(torch::Tensor x) {
    int n = x.numel();
    auto y = torch::empty_like(x);
    
    const int threads_per_block = 256;
    const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    selu_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

selu_cpp_source = "torch::Tensor selu_cuda(torch::Tensor x);"

# Compile the inline CUDA code for SELU
selu_op = load_inline(
    name="custom_selu",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_kernel_source,
    functions=["selu_cuda"],
    verbose=False,
    extra_cflags=["-D_FORCE_INLINES"],
    extra_cuda_cflags=["-lineinfo"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_cuda = selu_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu_cuda.selu_cuda(x)