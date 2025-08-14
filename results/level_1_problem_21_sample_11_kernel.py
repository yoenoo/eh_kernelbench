import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sigmoid_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_sigmoid_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

torch::Tensor elementwise_sigmoid_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    elementwise_sigmoid_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

sigmoid_cuda_header = "torch::Tensor elementwise_sigmoid_cuda(torch::Tensor x);"

sigmoid_cuda_module = load_inline(
    name="sigmoid_cuda",
    cpp_sources=sigmoid_cuda_header,
    cuda_sources=sigmoid_cuda_source,
    functions=["elementwise_sigmoid_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid_cuda = sigmoid_cuda_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid_cuda.elementwise_sigmoid_cuda(x)