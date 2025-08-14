import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

relu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(input[idx], 0.0f);
    }
}

torch::Tensor custom_relu_cuda(torch::Tensor input) {
    if (input.device().type() != torch::kCUDA) {
        AT_ERROR("Input tensor must be on CUDA device!");
    }

    auto input_ = input.contiguous();
    auto size = input_.numel();
    auto output = torch::empty_like(input_);
    
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    relu_kernel<<<num_blocks, block_size>>>(
        input_.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    cudaDeviceSynchronize();
    
    return output;
}
"""

relu_cuda_cpp_source = "torch::Tensor custom_relu_cuda(torch::Tensor input);"

custom_relu = load_inline(
    name="custom_relu",
    cpp_sources=relu_cuda_cpp_source,
    cuda_sources=relu_cuda_source,
    functions=["custom_relu_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_relu = custom_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_relu.custom_relu_cuda(x)