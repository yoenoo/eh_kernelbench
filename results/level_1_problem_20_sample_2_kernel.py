import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom LeakyReLU CUDA kernel implementation
leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void leaky_relu_forward_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output,
                                float negative_slope, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t val = input[idx];
        output[idx] = (val > 0.0f) ? val : val * negative_slope;
    }
}

torch::Tensor leaky_relu_forward_cuda(torch::Tensor input, float negative_slope) {
    auto output = torch::empty_like(input);
    int size = input.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "leaky_relu_forward", ([&] {
        leaky_relu_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), negative_slope, size);
    }));

    return output;
}
"""

leaky_relu_header = "torch::Tensor leaky_relu_forward_cuda(torch::Tensor input, float negative_slope);"

# Compile the CUDA kernel
leaky_relu_module = load_inline(
    name='leaky_relu_cuda',
    cpp_sources=leaky_relu_header,
    cuda_sources=leaky_relu_source,
    functions=['leaky_relu_forward_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=[''],
)

class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
        self.leaky_relu_forward = leaky_relu_module.leaky_relu_forward_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu_forward(x, self.negative_slope)