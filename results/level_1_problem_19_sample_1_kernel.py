import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

relu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
    }
}

torch::Tensor custom_relu_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    custom_relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    cudaDeviceSynchronize();
    return output;
}
"""

relu_cpp_source = "torch::Tensor custom_relu_forward(torch::Tensor input);"

relu_ops = load_inline(
    name="custom_relu",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_cuda_source,
    functions=["custom_relu_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu_op = relu_ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_op.custom_relu_forward(x)