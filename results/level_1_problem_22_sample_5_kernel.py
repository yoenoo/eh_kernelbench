import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for tanh
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
    auto output = torch::empty_like(x);
    int n = x.numel();
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    tanh_kernel<<<num_blocks, threads_per_block>>>(x.data_ptr<float>(), output.data_ptr<float>(), n);
    cudaDeviceSynchronize();
    return output;
}
"""

# Compile the inline CUDA code
tanh_cpp_source = "torch::Tensor tanh_cuda(torch::Tensor x);"

tanh_extension = load_inline(
    name="tanh_extension",
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_kernel_source,
    functions=["tanh_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh = tanh_extension.tanh_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh(x)