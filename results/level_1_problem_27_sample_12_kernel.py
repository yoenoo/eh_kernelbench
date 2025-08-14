import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for SELU activation function
selu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// SELU constants
const float alpha = 1.6732632423543772848170429916717f;
const float scale = 1.0507009873554804934193349852946f;

__global__ void selu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x > 0.f ? scale * x : scale * (alpha * expf(x) - alpha);
    }
}

torch::Tensor selu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    selu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

selu_cpp_source = "torch::Tensor selu_cuda(torch::Tensor input);"

# Compile the inline CUDA code for SELU activation
selu_cuda_op = load_inline(
    name="custom_selu",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=["selu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_cuda = selu_cuda_op  # Store the CUDA operator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu_cuda.selu_cuda(x)

# The get_inputs and get_init_inputs functions remain unchanged and are omitted here as per the problem statement.