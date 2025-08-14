import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* x, float* y, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sigmoid_x = 1.0f / (1.0f + expf(-x[idx]));
        y[idx] = x[idx] * sigmoid_x;
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    int n = x.numel();
    torch::Tensor y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}
"""

swish_header = "torch::Tensor swish_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Swish
swish_op = load_inline(
    name="swish_op",
    cpp_sources=swish_header,
    cuda_sources=swish_source,
    functions=["swish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.swish_op = swish_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish_op.swish_cuda(x)