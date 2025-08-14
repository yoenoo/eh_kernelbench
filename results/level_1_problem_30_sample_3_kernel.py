import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softsign_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softsign_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float abs_x = fabs(x[idx]);
        out[idx] = x[idx] / (1.0f + abs_x);
    }
}

torch::Tensor softsign_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    softsign_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

softsign_cpp_source = "torch::Tensor softsign_cuda(torch::Tensor x);"

softsign = load_inline(
    name="softsign",
    cpp_sources=softsign_cpp_source,
    cuda_sources=softsign_source,
    functions=["softsign_cuda"],
    verbose=False,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softsign = softsign

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softsign.softsign_cuda(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []