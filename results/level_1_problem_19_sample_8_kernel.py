import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(input[idx], static_cast<float>(0));
    }
}

torch::Tensor custom_relu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    custom_relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    cudaDeviceSynchronize();
    
    return output;
}
"""

relu_header = """
torch::Tensor custom_relu_cuda(torch::Tensor input);
"""

# Compile the custom ReLU kernel
custom_relu = load_inline(
    name="custom_relu",
    cpp_sources=relu_header,
    cuda_sources=relu_source,
    functions=["custom_relu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu = custom_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu.custom_relu_cuda(x)

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []