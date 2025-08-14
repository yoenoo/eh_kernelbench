import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for SELU activation
selu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define SELU_ALPHA 1.6732632423543772f
#define SELU_SCALE 1.0507009873554805f

__global__ void selu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        out[idx] = (xi >= 0.0f) ? SELU_SCALE * xi : SELU_ALPHA * (expf(xi) - 1.0f);
    }
}

torch::Tensor selu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    selu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

selu_cpp_source = "torch::Tensor selu_cuda(torch::Tensor x);"

# Compile the inline CUDA code
selu = load_inline(
    name="selu",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=["selu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu = selu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu.selu_cuda(x)