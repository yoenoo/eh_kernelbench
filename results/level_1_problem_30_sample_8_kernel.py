import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel and host function
softsign_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softsign_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xi = x[idx];
        out[idx] = xi / (1.0f + fabsf(xi));
    }
}

torch::Tensor softsign_cuda(torch::Tensor x) {
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto out = torch::empty_like(x, options);

    const int threads_per_block = 256;
    const int blocks_per_grid = (x.numel() + threads_per_block - 1) / threads_per_block;

    softsign_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), out.data_ptr<float>(), x.numel());

    return out;
}
"""

softsign_cuda_header = "torch::Tensor softsign_cuda(torch::Tensor x);"

# Compile the CUDA extension
softsign = load_inline(
    name='softsign',
    cuda_sources=softsign_cuda_source,
    cpp_sources=softsign_cuda_header,
    functions=['softsign_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softsign = softsign

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softsign.softsign_cuda(x)