import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for LeakyReLU
leaky_relu_cuda_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* x, float* out, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] > 0.f ? x[idx] : alpha * x[idx];
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor x, float alpha) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    leaky_relu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, size);
    return out;
}
"""

# Header for compilation
leaky_relu_header = "torch::Tensor leaky_relu_cuda(torch::Tensor x, float alpha);"

# Load the CUDA kernel
leaky_relu = load_inline(
    name="leaky_relu",
    cuda_sources=leaky_relu_cuda_src,
    cpp_sources=leaky_relu_header,
    functions=["leaky_relu_cuda"],
    verbose=True,
    extra_cflags=['-O3'],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
        self.leaky_relu = leaky_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu.leaky_relu_cuda(x, self.negative_slope)