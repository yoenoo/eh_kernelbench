import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus activation
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softplus_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = log(1 + exp(x[idx]));
    }
}

torch::Tensor softplus_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    softplus_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

softplus_cpp_source = "torch::Tensor softplus_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Softplus activation
softplus_extension = load_inline(
    name="softplus_extension",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softplus_cuda = softplus_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus_cuda.softplus_cuda(x)