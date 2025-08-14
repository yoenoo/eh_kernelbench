import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 4096
dim = 393216

# CUDA kernel definition for Softsign activation
softsign_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softsign_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        y[idx] = val / (1.0f + fabsf(val));
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

# Compile the custom CUDA kernel
softsign_mod = load_inline(
    name="softsign_mod",
    cpp_sources=softsign_cpp_source,
    cuda_sources=softsign_source,
    functions=["softsign_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softsign = softsign_mod  # Reference to the CUDA kernel module

    def forward(self, x):
        return self.softsign.softsign_cuda(x)

def get_inputs():
    # Generate input tensors on CUDA device
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []