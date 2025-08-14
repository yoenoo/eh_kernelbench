import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardSigmoid
hard_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>  // for std::max and std::min

__global__ void hard_sigmoid_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx] * (1.0f / 6.0f) + 0.5f;
        y[idx] = std::max(0.0f, std::min(1.0f, val));
    }
}

torch::Tensor hard_sigmoid_cuda(torch::Tensor x) {
    auto n = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    hard_sigmoid_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}
"""

hard_sigmoid_cpp_source = "torch::Tensor hard_sigmoid_cuda(torch::Tensor x);"

# Compile the inline CUDA code for HardSigmoid
hard_sigmoid = load_inline(
    name="hard_sigmoid",
    cpp_sources=hard_sigmoid_cpp_source,
    cuda_sources=hard_sigmoid_source,
    functions=["hard_sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hard_sigmoid = hard_sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hard_sigmoid.hard_sigmoid_cuda(x)