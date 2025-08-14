import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sigmoid_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = 1.0 / (1.0 + expf(-x[idx]));
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    sigmoid_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

sigmoid_cuda_header = "torch::Tensor sigmoid_cuda(torch::Tensor x);"

sigmoid_cuda_ops = load_inline(
    name="sigmoid_cuda",
    cpp_sources=sigmoid_cuda_header,
    cuda_sources=sigmoid_cuda_source,
    functions=["sigmoid_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid = sigmoid_cuda_ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid.sigmoid_cuda(x)

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []