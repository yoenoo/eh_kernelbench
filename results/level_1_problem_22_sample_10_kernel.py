import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tanh_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = tanhf(x[idx]);
    }
}

torch::Tensor tanh_cuda(torch::Tensor x) {
    int64_t n = x.numel();
    auto options = x.options();
    auto out = torch::empty({n}, options).view_as(x);  // preserve original shape

    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    tanh_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);

    return out;
}
"""

tanh_cpp_header = """
torch::Tensor tanh_cuda(torch::Tensor x);
"""

# Compile the custom CUDA kernel
tanh_cuda = load_inline(
    name="tanh_cuda",
    cpp_sources=tanh_cpp_header,
    cuda_sources=tanh_kernel_source,
    functions=["tanh_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh_cuda = tanh_cuda  # Access the loaded module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh_cuda.tanh_cuda(x)