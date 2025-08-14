import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tanh_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void tanh_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = tanhf(x[idx]);
    }
}

torch::Tensor tanh_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

tanh_kernel_cpp_header = "torch::Tensor tanh_cuda(torch::Tensor x);"

# Compile the custom tanh kernel
tanh_kernel = load_inline(
    name="tanh_kernel",
    cpp_sources=tanh_kernel_cpp_header,
    cuda_sources=tanh_kernel_source,
    functions=["tanh_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh = tanh_kernel  # Store the loaded kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh.tanh_cuda(x)