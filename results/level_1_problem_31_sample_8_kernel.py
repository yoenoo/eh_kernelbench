import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

elu_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void elu_kernel(const float* x, float* out, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
    }
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    int64_t size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    elu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, size);

    return out;
}
"""

elu_cpp_header = "torch::Tensor elu_cuda(torch::Tensor x, float alpha);"

# Compile the CUDA extension
elu = load_inline(
    name="elu",
    cpp_sources=elu_cpp_header,
    cuda_sources=elu_source,
    functions=["elu_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super(ModelNew, self).__init__()
        self.alpha = alpha
        self.elu_cuda_mod = elu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu_cuda_mod.elu_cuda(x, self.alpha)