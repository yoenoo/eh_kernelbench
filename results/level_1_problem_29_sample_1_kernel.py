import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus activation
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softplus_kernel(const float* x, float* y, const int n, const float beta, const float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float z = x[idx] * beta;
        y[idx] = (z > threshold) ? (z / beta) : (1.0 / beta * log(1.0 + exp(z)));
    }
}

torch::Tensor softplus_cuda(torch::Tensor x) {
    const float beta = 1.0;
    const float threshold = 20.0;

    auto n = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    softplus_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n, beta, threshold
    );

    return y;
}
"""

softplus_cpp_source = "torch::Tensor softplus_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Softplus activation
softplus_module = load_inline(
    name="softplus_cuda",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus_cuda = softplus_module

    def forward(self, x):
        return self.softplus_cuda.softplus_cuda(x)