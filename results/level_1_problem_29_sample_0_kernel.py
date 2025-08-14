import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softplus_kernel(const float* x, float* y, int size, float beta=1.0f, float threshold=20.0f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float tmp = beta * x[idx];
        y[idx] = (tmp > threshold) ? tmp : (1.0f / beta) * logf(1.0f + expf(tmp));
    }
}

torch::Tensor softplus_cuda(torch::Tensor x, float beta=1.0, float threshold=20.0) {
    auto size = x.numel();
    auto y = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    softplus_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size, beta, threshold);
    
    return y;
}
"""

softplus_cpp_source = "torch::Tensor softplus_cuda(torch::Tensor x, float beta=1.0, float threshold=20.0);"

# Compile the inline CUDA code
softplus = load_inline(
    name="softplus",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softplus = softplus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus.softplus_cuda(x)