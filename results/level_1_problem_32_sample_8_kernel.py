import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void hardtanh_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = fmaxf(-1.0f, fminf(1.0f, val));
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    hardtanh_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_hardtanh_cpp_source = "torch::Tensor hardtanh_cuda(torch::Tensor x);"

hardtanh_extension = load_inline(
    name="hardtanh_extension",
    cpp_sources=elementwise_hardtanh_cpp_source,
    cuda_sources=elementwise_hardtanh_source,
    functions=["hardtanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardtanh = hardtanh_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardtanh.hardtanh_cuda(x)