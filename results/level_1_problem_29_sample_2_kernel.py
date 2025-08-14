import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus activation
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softplus_kernel(const float* x, float* out, int size, float beta=1.0f, float threshold=20.0f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float yi = x[idx] * beta;
        out[idx] = (yi > threshold) ? (yi / beta) : (1.0f / beta * log(1.0f + exp(yi)));
    }
}

torch::Tensor softplus_cuda(torch::Tensor x, float beta=1.0, float threshold=20.0) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    softplus_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size, 
        beta, 
        threshold
    );

    return out;
}
"""

softplus_cpp_source = "torch::Tensor softplus_cuda(torch::Tensor x, float beta=1.0, float threshold=20.0);"

# Compile the inline CUDA code for Softplus activation
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
        super().__init__()
        self.beta = 1.0
        self.threshold = 20.0
        self.softplus_cuda = softplus

    def forward(self, x):
        return self.softplus_cuda.softplus_cuda(x, self.beta, self.threshold)