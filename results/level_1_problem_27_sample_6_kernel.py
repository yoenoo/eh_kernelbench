import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_selu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_selu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        if (val > 0.0f) {
            out[idx] = 1.05070098f * val;
        } else {
            out[idx] = 1.05070098f * 1.67326324f * (expf(val) - 1.0f);
        }
    }
}

torch::Tensor elementwise_selu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_selu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size
    );
    return out;
}
"""

elementwise_selu_cpp_source = (
    "torch::Tensor elementwise_selu_cuda(torch::Tensor x);"
)

elementwise_selu = load_inline(
    name="elementwise_selu",
    cpp_sources=elementwise_selu_cpp_source,
    cuda_sources=elementwise_selu_source,
    functions=["elementwise_selu_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.elementwise_selu = elementwise_selu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elementwise_selu.elementwise_selu_cuda(x)