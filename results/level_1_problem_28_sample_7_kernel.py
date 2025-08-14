import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_hardsigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardsigmoid_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx] + 3.0f;
        val = fmaxf(0.0f, val);
        val = fminf(6.0f, val);
        y[idx] = val * (1.0f / 6.0f);
    }
}

torch::Tensor elementwise_hardsigmoid_cuda(torch::Tensor x) {
    int64_t numel = x.numel();
    auto y = torch::empty_like(x);
    
    const int block_size = 256;
    int num_blocks = (numel + block_size - 1) / block_size;

    hardsigmoid_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), numel);

    return y;
}
"""

elementwise_hardsigmoid_cpp_source = (
    "torch::Tensor elementwise_hardsigmoid_cuda(torch::Tensor x);"
)

elementwise_hardsigmoid = load_inline(
    name="elementwise_hardsigmoid",
    cuda_sources=elementwise_hardsigmoid_source,
    cpp_sources=elementwise_hardsigmoid_cpp_source,
    functions=["elementwise_hardsigmoid_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardsigmoid_func = elementwise_hardsigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardsigmoid_func.elementwise_hardsigmoid_cuda(x)