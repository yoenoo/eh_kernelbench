import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for Swish activation (x * sigmoid(x))
swish_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_forward_kernel(const float* x, float* y, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sigmoid_x = 1.0f / (1.0f + expf(-x[idx]));
        y[idx] = x[idx] * sigmoid_x;
    }
}

torch::Tensor swish_forward_cuda(torch::Tensor x) {
    int n = x.numel();
    torch::Tensor y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    swish_forward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    cudaDeviceSynchronize();

    return y;
}
"""

swish_kernel_header = """
torch::Tensor swish_forward_cuda(torch::Tensor x);
"""

# Compile the CUDA extension inline
swish_ops = load_inline(
    name='swish_cuda',
    cpp_sources=swish_kernel_header,
    cuda_sources=swish_kernel_source,
    functions=['swish_forward_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.swish_forward = swish_ops.swish_forward_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish_forward(x)