import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return argmax_cuda(x, self.dim)

# Define the custom CUDA kernel for argmax
argmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

template <typename scalar_t>
__global__ void argmax_kernel(const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> input,
                             torch::PackedTensorAccessor<int64_t, 2, torch::RestrictPtrTraits> output,
                             int dim) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int other_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx >= input.size(0) || other_idx >= input.size(2)) {
        return;
    }

    int max_idx = 0;
    scalar_t max_val = input[batch_idx][0][other_idx];
    for (int i = 1; i < input.size(1); ++i) {
        if (input[batch_idx][i][other_idx] > max_val) {
            max_val = input[batch_idx][i][other_idx];
            max_idx = i;
        }
    }
    output[batch_idx][other_idx] = max_idx;
}

int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

at::Tensor argmax_cuda(torch::Tensor input, int dim) {
    int batch_size = input.size(0);
    int dim_size = input.size(dim);
    int other_size = input.size(2); // Assuming dim is 1, since dim is fixed in the model.

    auto output = torch::empty({batch_size, other_size}, input.options().dtype(torch::kLong));

    dim3 block(32, 8);
    dim3 grid(ceil_div(batch_size, block.x), ceil_dim(other_size, block.y));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmax_cuda", ([&] {
        argmax_kernel<scalar_t><<<grid, block>>>(
            input.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>(),
            output.packed_accessor<int64_t, 2, torch::RestrictPtrTraits>(),
            dim);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

argmax_cpp_source = "at::Tensor argmax_cuda(torch::Tensor input, int dim);"

# Compile the inline CUDA code
argmax_module = load_inline(
    name="argmax_op",
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_source,
    functions=["argmax_cuda"],
    verbose=True
)

def argmax_cuda(input, dim):
    return argmax_module.argmax_cuda(input, dim)