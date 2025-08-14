import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void elu_kernel(const float* x, float* out, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = val > 0.0f ? val : alpha * (expf(val) - 1.0f);
    }
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, size);

    return out;
}
"""

elu_cpp_source = "torch::Tensor elu_cuda(torch::Tensor x, float alpha);"

elu_cuda_module = load_inline(
    name="elu_cuda",
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_source,
    functions=["elu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return elu_cuda_module.elu_cuda(x, self.alpha)