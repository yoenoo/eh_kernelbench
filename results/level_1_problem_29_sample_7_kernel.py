import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softplus_kernel(const float* input, float* output, int size, float beta=1, float threshold=20) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = input[idx] * beta;
        output[idx] = (z > threshold) ? z / beta : (1.0 / beta) * log(1 + exp(z));
    }
}

torch::Tensor softplus_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    softplus_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

softplus_cpp_source = "torch::Tensor softplus_cuda(torch::Tensor input);"

# Compile the inline CUDA code for Softplus
softplus = load_inline(
    name="softplus",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softplus = softplus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus.softplus_cuda(x)