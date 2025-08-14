import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

swish_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float exp_val = expf(-x[idx]);
        float sigmoid = 1.0f / (1.0f + exp_val);
        y[idx] = x[idx] * sigmoid;
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

swish_cuda_header = "torch::Tensor swish_cuda(torch::Tensor x);"

swish_cuda = load_inline(
    name="swish_cuda",
    cpp_sources=swish_cuda_header,
    cuda_sources=swish_cuda_source,
    functions=["swish_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.swish = swish_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish.swish_cuda(x)