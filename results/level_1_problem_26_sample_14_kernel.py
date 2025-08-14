import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xv = x[idx];
        float x3 = xv * xv * xv;
        float temp = xv + 0.044715f * x3;
        const float sqrt_2_pi = 0.7978845608f;
        float tanh_val = tanhf(sqrt_2_pi * temp);
        out[idx] = 0.5f * xv * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    int size = x.numel();
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    
    return out;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"

gelu = load_inline(
    name="custom_gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu.gelu_cuda(x)