import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* x, float* out, int size) {
    const float sqrt_2_over_pi = 0.7978845608;
    const float coefficient = 0.044715;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float inner = sqrt_2_over_pi * (xi + coefficient * pow(xi, 3));
        out[idx] = 0.5 * xi * (1.0 + tanh(inner));
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

gelu_cpp_source = (
    "torch::Tensor gelu_cuda(torch::Tensor x);"
)

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
        super(ModelNew, self).__init__()
        self.gelu = gelu

    def forward(self, x):
        return self.gelu.gelu_cuda(x)