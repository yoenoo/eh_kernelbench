import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ELU activation
elu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void elu_kernel(const float* x, float* y, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        y[idx] = (val > 0) ? val : alpha * (expf(val) - 1);
    }
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), alpha, size);

    return y;
}
"""

elu_cpp_source = "torch::Tensor elu_cuda(torch::Tensor x, float alpha);"

# Compile the inline CUDA code for ELU
elu = load_inline(
    name="elu_extension",
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_source,
    functions=["elu_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
        self.elu = elu  # Store the loaded CUDA module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu.elu_cuda(x, self.alpha)