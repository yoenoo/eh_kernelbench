import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softsign_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softsign_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xi = x[idx];
        y[idx] = xi / (1.0f + fabsf(xi));
    }
}

torch::Tensor softsign_cuda(torch::Tensor x) {
    auto output = torch::empty_like(x);
    int64_t n = x.numel();
    const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    softsign_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

softsign_cpp_source = "torch::Tensor softsign_cuda(torch::Tensor x);"

# Compile the CUDA kernel
softsign = load_inline(
    name="softsign",
    cpp_sources=softsign_cpp_source,
    cuda_sources=softsign_source,
    functions=["softsign_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softsign = softsign

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softsign.softsign_cuda(x)