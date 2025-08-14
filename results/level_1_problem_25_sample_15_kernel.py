import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

swish_cuda_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void swish_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xi = x[idx];
        float exp_val = expf(-xi);
        float sigmoid_val = 1.0f / (1.0f + exp_val);
        y[idx] = xi * sigmoid_val;
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto n = x.numel();
    auto y = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

swish_cuda_header = "torch::Tensor swish_cuda(torch::Tensor x);"

swish_cu = load_inline(
    name="swish_cuda",
    cpp_sources=swish_cuda_header,
    cuda_sources=swish_cuda_source,
    functions=["swish_cuda"],
    verbose=True,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.swish_cu = swish_cu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish_cu.swish_cuda(x)