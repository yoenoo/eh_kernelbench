import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softplus_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = std::log(1.0f + std::exp(x[idx]));
    }
}

torch::Tensor softplus_cuda(torch::Tensor x) {
    int n = x.numel();
    auto y = torch::empty_like(x);
    
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    softplus_kernel<<<grid_size, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

softplus_h_source = "torch::Tensor softplus_cuda(torch::Tensor x);"

# Compile the inline CUDA code
softplus = load_inline(
    name="softplus",
    cpp_sources=softplus_h_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softplus_op = softplus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus_op.softplus_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []