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
        float xi = x[idx];
        const float a = 0.044715f;
        const float k = sqrtf(2.0f / M_PI);
        float tanh_term = tanhf(k * (xi + a * xi * xi * xi));
        out[idx] = 0.5f * xi * (1.0f + tanh_term);
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto x_gpu = x.cuda();
    auto size = x_gpu.numel();
    auto out = torch::empty_like(x_gpu);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(
        x_gpu.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );

    return out;
}
"""

gelu_cpp_source = """
torch::Tensor gelu_cuda(torch::Tensor x);
"""

gelu_module = load_inline(
    name="gelu_cuda",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = gelu_module

    def forward(self, x):
        x = x.cuda()
        return self.gelu.gelu_cuda(x)