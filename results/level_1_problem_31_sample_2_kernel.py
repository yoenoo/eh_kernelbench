import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

elu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elu_kernel(const float* input, float* output, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = (val > 0) ? val : alpha * (exp(val) - 1);
    }
}

torch::Tensor elu_cuda(torch::Tensor input, float alpha) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    elu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), alpha, size);
    cudaDeviceSynchronize();
    return output;
}
"""

elu_cpp_source = "torch::Tensor elu_cuda(torch::Tensor input, float alpha);"

# Compile the inline CUDA code for ELU
elu_cuda_op = load_inline(
    name="elu_cuda_op",
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_source,
    functions=["elu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
        self.elu_cuda_op = elu_cuda_op  # Reference the custom CUDA operator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu_cuda_op.elu_cuda(x, self.alpha)