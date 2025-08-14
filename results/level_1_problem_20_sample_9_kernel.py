import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float slope, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : val * slope;
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor input, float slope) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    leaky_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        slope,
        size
    );

    return output;
}
"""

leaky_relu_cpp_src = (
    "torch::Tensor leaky_relu_cuda(torch::Tensor input, float slope);"
)

leaky_relu_extension = load_inline(
    name="leaky_relu",
    cuda_sources=leaky_relu_source,
    cpp_sources=leaky_relu_cpp_src,
    functions=["leaky_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
        self.cuda_leaky_relu = leaky_relu_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cuda_leaky_relu.leaky_relu_cuda(x, self.negative_slope)