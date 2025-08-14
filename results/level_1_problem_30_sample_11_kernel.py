import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softsign_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softsign_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = val / (1.0f + fabsf(val));
    }
}

torch::Tensor softsign_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    softsign_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    cudaDeviceSynchronize();
    return out;
}
"""

softsign_header = """
torch::Tensor softsign_cuda(torch::Tensor x);
"""

softsign_module = load_inline(
    name="softsign",
    cpp_sources=softsign_header,
    cuda_sources=softsign_source,
    functions=["softsign_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softsign = softsign_module.softsign_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softsign(x)