import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for SELU activation
selu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// SELU constants
#define SEL_ALPHA 1.6732632423543772848170429916717f
#define SEL_SCALE 1.0507009873554804934193349852946f

__global__ void selu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x > 0 ? SEL_SCALE * x : SEL_SCALE * SEL_ALPHA * (exp(x) - 1);
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

selu_header = "torch::Tensor selu_cuda(torch::Tensor input);"

# Compile the inline CUDA code for SELU
selu = load_inline(
    name="selu",
    cpp_sources=selu_header,
    cuda_sources=selu_source,
    functions=["selu_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_cuda = selu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu_cuda.selu_cuda(x)

# Keep the input generation functions as given
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []