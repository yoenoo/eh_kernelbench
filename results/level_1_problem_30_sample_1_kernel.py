import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softsign activation
softsign_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softsign_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        y[idx] = xi / (1.0f + fabsf(xi));
    }
}

torch::Tensor softsign_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    softsign_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

softsign_cpp_source = "torch::Tensor softsign_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Softsign activation
softsign = load_inline(
    name="softsign",
    cpp_sources=softsign_cpp_source,
    cuda_sources=softsign_source,
    functions=["softsign_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softsign = softsign

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softsign.softsign_cuda(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')  # Move data to GPU for direct processing
    return [x]

def get_init_inputs():
    return []