import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void custom_sigmoid_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

torch::Tensor custom_sigmoid_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    custom_sigmoid_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

sigmoid_cpp_source = "torch::Tensor custom_sigmoid_cuda(torch::Tensor x);"

custom_sigmoid = load_inline(
    name="custom_sigmoid",
    cpp_sources=sigmoid_cpp_source,
    cuda_sources=sigmoid_source,
    functions=["custom_sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = custom_sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid.custom_sigmoid_cuda(x)