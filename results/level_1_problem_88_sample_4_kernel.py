import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

gelu_approx_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_approx_kernel(const float* x, float* out, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float inner = sqrt(2.0f / M_PI) * (xi + 0.044715f * xi * xi * xi);
        out[idx] = 0.5f * xi * (1.0f + tanhf(inner));
    }
}

torch::Tensor gelu_approx_cuda(torch::Tensor x) {
    int size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_approx_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

gelu_approx_cpp_source = "torch::Tensor gelu_approx_cuda(torch::Tensor x);"

gelu_approx = load_inline(
    name="gelu_approx",
    cpp_sources=gelu_approx_cpp_source,
    cuda_sources=gelu_approx_source,
    functions=["gelu_approx_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_approx = gelu_approx

    def forward(self, x):
        return self.gelu_approx.gelu_approx_cuda(x)

def get_inputs():
    return [torch.rand(batch_size, dim, device="cuda")]

def get_init_inputs():
    return []