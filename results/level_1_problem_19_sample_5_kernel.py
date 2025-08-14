import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

relu_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(input[idx], 0.f);
    }
}

torch::Tensor custom_relu(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    custom_relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

relu_cpp_source = "torch::Tensor custom_relu(torch::Tensor input);"

custom_relu_op = load_inline(
    name="custom_relu",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_kernel_source,
    functions=["custom_relu"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_relu = custom_relu_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_relu.custom_relu(x)