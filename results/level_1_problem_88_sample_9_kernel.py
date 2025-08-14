import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x_val = x[idx];
    float x_cubed = x_val * x_val * x_val;
    const float sqrt_2_over_pi = 0.79788456f;
    float inner = sqrt_2_over_pi * (x_val + 0.044715f * x_cubed);
    float tanh_val = tanhf(inner);
    out[idx] = 0.5f * x_val * (1.0f + tanh_val);
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    const int threads = 256;
    int64_t numel = x.numel();
    auto out = torch::empty_like(x);
    const int blocks = (numel + threads - 1) / threads;

    gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), numel);
    return out;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"

gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu_cuda = gelu

    def forward(self, x):
        return self.gelu_cuda.gelu_cuda(x)