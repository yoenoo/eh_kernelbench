import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sigmoid_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

torch::Tensor sigmoid_forward(torch::Tensor x) {
    const auto n = x.numel();
    auto y = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    sigmoid_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

sigmoid_kernel_h_source = "torch::Tensor sigmoid_forward(torch::Tensor x);"

sigmoid_cuda = load_inline(
    name='sigmoid_cuda',
    cuda_sources=sigmoid_kernel_source,
    cpp_sources=sigmoid_kernel_h_source,
    functions=['sigmoid_forward'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid_cuda = sigmoid_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid_cuda.sigmoid_forward(x)