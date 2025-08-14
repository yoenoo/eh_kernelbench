import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        y[idx] = tanh(xi);
    }
}

torch::Tensor tanh_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

tanh_cpp_source = "torch::Tensor tanh_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Tanh activation
tanh_op = load_inline(
    name="tanh_op",
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_source,
    functions=["tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh_cuda_op = tanh_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh_cuda_op.tanh_cuda(x)