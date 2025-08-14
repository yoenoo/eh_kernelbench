import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softsign_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softsign_kernel(const float* x_data, float* out_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x_data[idx];
        float denom = 1.0f + fabsf(val);
        out_data[idx] = val / denom;
    }
}

torch::Tensor softsign_cuda(torch::Tensor x) {
    auto n = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    softsign_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);

    return out;
}
"""

softsign_cpp = "torch::Tensor softsign_cuda(torch::Tensor x);"

softsign = load_inline(
    name="softsign",
    cpp_sources=softsign_cpp,
    cuda_sources=softsign_source,
    functions=["softsign_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softsign = softsign

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softsign.softsign_cuda(x)