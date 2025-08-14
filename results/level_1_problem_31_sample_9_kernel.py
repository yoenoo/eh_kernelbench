import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ELU activation
elu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void elu_kernel(const float* x, float* out, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = val > 0.f ? val : alpha * (expf(val) - 1.f);
    }
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    const int threads_per_block = 256;
    const int blocks_per_grid = (x.numel() + threads_per_block - 1) / threads_per_block;

    auto out = torch::empty_like(x);
    
    elu_kernel<<<blocks_per_grid, threads_per_block>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        alpha,
        x.numel()
    );

    return out;
}
"""

elu_cpp_source = "torch::Tensor elu_cuda(torch::Tensor x, float alpha);"

# Compile the inline CUDA code for ELU
elu = load_inline(
    name="elu_cuda",
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_cuda_source,
    functions=["elu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
        self.elu_cuda = elu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu_cuda.elu_cuda(x, self.alpha)