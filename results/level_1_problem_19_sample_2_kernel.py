import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

relu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    cudaDeviceSynchronize();
    
    return output;
}
"""

relu_cpp_source = "torch::Tensor relu_cuda(torch::Tensor input);"

relu_op = load_inline(
    name="custom_relu",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_cuda_source,
    functions=["relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu = relu_op
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu.relu_cuda(x)