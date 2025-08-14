import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

relu_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
    }
}

torch::Tensor custom_relu(torch::Tensor input) {
    int elements = input.numel();
    torch::Tensor output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (elements + block_size - 1) / block_size;

    custom_relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), 
                                                  output.data_ptr<float>(), 
                                                  elements);

    return output;
}
"""

cpp_src = "torch::Tensor custom_relu(torch::Tensor input);"

custom_relu_ext = load_inline(
    name="custom_relu",
    cpp_sources=cpp_src,
    cuda_sources=relu_kernel,
    functions=["custom_relu"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_op = custom_relu_ext

    def forward(self, x):
        return self.relu_op.custom_relu(x)