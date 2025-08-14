import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = max(0.f, x[idx]);
    }
}

torch::Tensor relu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    relu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

relu_header = "torch::Tensor relu_cuda(torch::Tensor x);"

relu_module = load_inline(
    name='relu_op',
    cpp_sources=relu_header,
    cuda_sources=relu_source,
    functions=['relu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu = relu_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu.relu_cuda(x)