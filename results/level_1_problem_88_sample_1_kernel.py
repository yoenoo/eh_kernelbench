import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Custom GELU kernel implementation
gelu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ 
void gelu_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x_val = x[idx];
        float x_pow3 = x_val * x_val * x_val;
        float inner = (x_val + 0.044715 * x_pow3) * sqrt(2.0f / M_PI);
        float tanh_val = tanh(inner);
        y[idx] = 0.5f * x_val * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    int total = x.numel();
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    auto y = torch::empty_like(x);
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), total);
    
    return y;
}
"""

# Declare function prototypes
gelu_cuda_source_declaration = """
torch::Tensor gelu_cuda(torch::Tensor x);
"""

# Compile the custom CUDA kernel
gelu = load_inline(
    name="gelu_cuda",
    cpp_sources=gelu_cuda_source_declaration,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_cuda = gelu

    def forward(self, x):
        # Replace the original GELU computation with custom CUDA kernel
        return self.gelu_cuda.gelu_cuda(x)

# Update input functions with cuda support
def get_inputs():
    return [torch.rand(batch_size, dim, device='cuda')]

def get_init_inputs():
    return []