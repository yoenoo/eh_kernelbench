import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for SELU activation
selu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void selu_kernel(const float* input, float* output, int size) {
    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x > 0.f ? scale * x : scale * alpha * (exp(x) - 1.f);
    }
}

torch::Tensor selu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    selu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    cudaDeviceSynchronize();
    return output;
}
"""

selu_cpp_source = "torch::Tensor selu_cuda(torch::Tensor input);"

# Compile the inline CUDA code for SELU activation
selu = load_inline(
    name="custom_selu",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=["selu_cuda"],
    verbose=True,
    extra_cflags=["-g", "-Wno-reorder"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_cuda = selu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu_cuda.selu_cuda(x)