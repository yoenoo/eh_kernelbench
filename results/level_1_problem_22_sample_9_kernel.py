import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* in_data, float* out_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = in_data[idx];
        out_data[idx] = __tanhf(x); // Using CUDA's intrinsic for float tanh
    }
}

torch::Tensor tanh_cuda(torch::Tensor in) {
    auto size = in.numel();
    auto out = torch::empty_like(in);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_kernel<<<num_blocks, block_size>>>(in.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

tanh_cpp_source = "torch::Tensor tanh_cuda(torch::Tensor in);"

# Compile the inline CUDA code
tanh_op = load_inline(
    name="tanh_op",
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_source,
    functions=["tanh_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh_cuda_op = tanh_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh_cuda_op.tanh_cuda(x)