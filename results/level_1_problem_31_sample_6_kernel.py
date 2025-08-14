import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ELU activation
elu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void elu_kernel(const float* __restrict__ x, float* __restrict__ out, const float alpha, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (x[idx] > 0) ? x[idx] : alpha * (exp(x[idx]) - 1);
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

# Compile the inline CUDA code for ELU activation
elu_module = load_inline(
    name="elu_module",
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_source,
    functions=["elu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
        self.elu = elu_module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu.elu_cuda(x, self.alpha)

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return [1.0]  # Provide alpha value for initialization