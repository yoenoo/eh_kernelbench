import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for Tanh
tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

torch::Tensor tanh_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

tanh_cpp_source = """
#include <torch/extension.h>

extern "C" {
    torch::Tensor tanh_cuda(torch::Tensor input);
}
"""

# Compile the inline CUDA code
tanh_cuda = load_inline(
    name="tanh_cuda",
    cpp_sources=[tanh_cpp_source],
    cuda_sources=[tanh_source],
    functions=["tanh_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh_cuda = tanh_cuda  # This holds the module for the extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh_cuda.tanh_cuda(x)