import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float y_i = 0.5f * xi * (1.0f + erf(xi / M_SQRT2));
        y[idx] = y_i;
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto output = torch::empty_like(x);
    int size = x.numel();
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

gelu_cpp_src = "torch::Tensor gelu_cuda(torch::Tensor x);"

gelu_extension = load_inline(
    name='gelu_op',
    cpp_sources=gelu_cpp_src,
    cuda_sources=gelu_source,
    functions=['gelu_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu = gelu_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu.gelu_cuda(x)