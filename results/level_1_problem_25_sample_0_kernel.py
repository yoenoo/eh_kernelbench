import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void swish_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sig = 1.0 / (1.0 + expf(-x[idx]));
        y[idx] = x[idx] * sig;
    }
}

torch::Tensor swish_forward_cuda(torch::Tensor x) {
    int n = x.numel();
    auto y = torch::empty_like(x);
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    swish_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    cudaDeviceSynchronize();
    return y;
}
"""

swish_h_source = "torch::Tensor swish_forward_cuda(torch::Tensor x);"

# Compile the CUDA kernel
swish_extension = load_inline(
    name="swish_cuda",
    cpp_sources=swish_h_source,
    cuda_sources=swish_source,
    functions=["swish_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.swish_forward = swish_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish_forward.swish_forward_cuda(x)