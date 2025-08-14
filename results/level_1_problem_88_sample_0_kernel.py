import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* x, float* out, int size) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float constant = 0.044715f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float tanh_val = tanhf(sqrt_2_over_pi * (xi + constant * xi * xi * xi));
        out[idx] = 0.5f * xi * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    
    return out;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"

# Compile the inline CUDA code for GELU
gelu_extension = load_inline(
    name="gelu_cuda",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_cuda = gelu_extension

    def forward(self, x):
        return self.gelu_cuda.gelu_cuda(x)

def get_inputs():
    return [torch.rand(batch_size, dim).cuda()]

def get_init_inputs():
    return []

batch_size = 8192
dim = 8192