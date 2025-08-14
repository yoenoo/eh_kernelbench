import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for SELU activation
selu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// SELU constants
constexpr float alpha = 1.6732632423543772848170429916717f;
constexpr float scale = 1.0507009873554804934193349852946f;

__global__ void selu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val > 0 ? scale * val : scale * (alpha * exp(val) - alpha);
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

# Compile the inline CUDA code for SELU
selu = load_inline(
    name="selu",
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
        self.selu_cuda = selu  # Reference to the custom CUDA kernel function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the custom CUDA kernel for SELU
        return self.selu_cuda.selu_cuda(x)

# Input generation remains unchanged as per original architecture
def get_inputs():
    x = torch.rand(batch_size, dim).cuda()  # Move input to GPU
    return [x]

def get_init_inputs():
    return []