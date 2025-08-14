import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

gelu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(torch::PackedTensorAccessor<float,2> x, torch::PackedTensorAccessor<float,2> out, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim) {
        const int row = idx / dim;
        const int col = idx % dim;
        float xx = x[row][col];
        float y = 0.5f * xx * (1.0f + tanh(sqrt(2.0f / M_PI) * (xx + 0.044715f * pow(xx,3))));
        out[row][col] = y;
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    auto out = torch::empty_like(x);
    int size = batch_size * dim;
    const int block_size = 256;
    const int num_blocks = (size + block_size -1)/block_size;
    gelu_kernel<<<num_blocks, block_size>>>(x.packed_accessor<float,2>(), out.packed_accessor<float,2>(), batch_size, dim);
    return out;
}
"""

gelu_cuda_header = "torch::Tensor gelu_cuda(torch::Tensor x);"

gelu_cuda_ext = load_inline(
    name="gelu_cuda",
    cpp_sources=gelu_cuda_header,
    cuda_sources=gelu_cuda_source,
    functions=["gelu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_cuda = gelu_cuda_ext

    def forward(self, x):
        return self.gelu_cuda.gelu_cuda(x)

batch_size = 8192
dim = 8192

def get_inputs():
    return [torch.rand(batch_size, dim).cuda()]

def get_init_inputs():
    return []