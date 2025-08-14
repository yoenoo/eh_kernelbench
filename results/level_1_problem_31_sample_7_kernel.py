import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

elu_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void elu_kernel(const float* x, float* y, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float value = x[idx];
        y[idx] = (value > 0.0f) ? value : alpha * (expf(value) - 1.0f);
    }
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    int size = x.numel();
    torch::Tensor y = torch::empty_like(x);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    elu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), alpha, size);

    return y;
}
"""

elu_kernel_header = "torch::Tensor elu_cuda(torch::Tensor x, float alpha);"

elu_op = load_inline(
    name="elu_op",
    cpp_sources=elu_kernel_header,
    cuda_sources=elu_kernel_source,
    functions=["elu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
        self.elu_cuda = elu_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu_cuda.elu_cuda(x, self.alpha)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return [1.0]