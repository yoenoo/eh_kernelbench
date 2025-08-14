import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom GELU CUDA kernel
gelu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void gelu_forward_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const scalar_t x = input[idx];
        const scalar_t y = x * 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
        output[idx] = y;
    }
}

at::Tensor gelu_forward_cuda(const at::Tensor& input) {
    const auto n = input.numel();
    auto output = at::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "gelu_forward_cuda", [&] {
        gelu_forward_kernel<scalar_t><<<num_blocks, block_size, 0, input.type().device().cuda_stream()>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            n
        );
    });

    return output;
}
"""

# Compile the inline CUDA code
gelu_module = load_inline(
    name='gelu_cuda',
    cpp_sources='',
    cuda_sources=gelu_cuda_source,
    functions=['gelu_forward_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_forward = gelu_module.gelu_forward_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu_forward(x)