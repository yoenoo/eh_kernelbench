import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for SELU
selu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define SELU_ALPHA 1.6732632423543772848170429916717f
#define SELU_SCALE 1.0507009873554804934193349852946f

__global__ void selu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x >= 0 ? SELU_SCALE * x : SELU_SCALE * SELU_ALPHA * (expf(x) - 1);
    }
}

torch::Tensor selu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    selu_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    cudaDeviceSynchronize();
    return output;
}
"""

selu_cuda_header = "torch::Tensor selu_cuda(torch::Tensor input);"

# Compile the inline CUDA code
selu_cuda = load_inline(
    name="selu_cuda",
    cpp_sources=selu_cuda_header,
    cuda_sources=selu_cuda_source,
    functions=["selu_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_cuda = selu_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu_cuda.selu_cuda(x)