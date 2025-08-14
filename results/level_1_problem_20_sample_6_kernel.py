import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for LeakyReLU
leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void leaky_relu_kernel(const scalar_t* __restrict__ input,
                                scalar_t* __restrict__ output,
                                const float negative_slope,
                                const size_t n) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        scalar_t val = input[index];
        output[index] = (val > 0.0) ? val : val * negative_slope;
    }
}

int leaky_relu_cuda(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const int n = input.numel();
    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "leaky_relu_cuda", ([&] {
        leaky_relu_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            negative_slope,
            n);
    }));

    return 1;
}
"""

leaky_relu_cpp_source = "int leaky_relu_cuda(torch::Tensor input, torch::Tensor output, float negative_slope);"

# Compile the CUDA extension
leaky_relu = load_inline(
    name="leaky_relu",
    cpp_sources=[leaky_relu_cpp_source],
    cuda_sources=[leaky_relu_source],
    functions=["leaky_relu_cuda"],
    verbose=False,
    extra_cflags=["-DDEBUG"],
    extra_cuda_cflags=["-lineinfo"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
        self.fused_leaky_relu = leaky_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        self.fused_leaky_relu.leaky_relu_cuda(x, output, self.negative_slope)
        return output

# Keep the input generation same
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda")  # Moved to GPU for better performance
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed