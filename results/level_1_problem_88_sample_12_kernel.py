import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

gelu_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS 256

__global__ void fused_gelu_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = x[idx];
        float inner = sqrt(2.0f / M_PI) * (x_val + 0.044715f * pow(x_val, 3.0f));
        out[idx] = 0.5f * x_val * (1.0f + tanhf(inner));
    }
}

torch::Tensor fused_gelu(const torch::Tensor& x) {
    int size = x.numel();
    const int blocks = (size + THREADS - 1) / THREADS;

    auto out = torch::empty_like(x);
    fused_gelu_kernel<<<blocks, THREADS>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""

gelu_cpp_source = "torch::Tensor fused_gelu(const torch::Tensor& x);"

gelu_op = load_inline(
    name="fused_gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_cuda_source,
    functions=["fused_gelu"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.fused_gelu = gelu_op

    def forward(self, x):
        return self.fused_gelu.fused_gelu(x)