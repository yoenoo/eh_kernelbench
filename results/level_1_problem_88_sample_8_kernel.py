import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

gelu_custom_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void custom_gelu_kernel(const float* x, float* out, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float tanh_part = tanh(xi + 0.044715 * xi * xi * xi) * sqrt(2.0 / M_PI);
        out[idx] = 0.5 * xi * (1.0 + tanh_part);
    }
}

torch::Tensor gelu_forward_cuda(torch::Tensor x) {
    const auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    custom_gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""

gelu_cpp_source = "torch::Tensor gelu_forward_cuda(torch::Tensor x);"

gelu_cu = load_inline(name="gelu_cu",
                   cuda_sources=gelu_custom_source,
                   cpp_sources=gelu_cpp_source,
                   functions=["gelu_forward_cuda"],
                   verbose=True)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_gelu = gelu_cu

    def forward(self, x):
        return self.custom_gelu.gelu_forward_cuda(x)