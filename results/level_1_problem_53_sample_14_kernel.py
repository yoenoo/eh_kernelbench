import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for min reduction
min_reduction_source = """
#include <torch/threads.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void min_kernel(const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> input,
                          torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits> output,
                          int dim1, int dim2, int outer_dim, int inner_dim) {
    int outer = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer >= outer_dim) return;

    scalar_t min_val = std::numeric_limits<scalar_t>::max();
    for (int i = 0; i < inner_dim; ++i) {
        scalar_t val = input[outer][i];
        if (val < min_val) {
            min_val = val;
        }
    }
    output[outer] = min_val;
}

torch::Tensor min_reduction_cuda(torch::Tensor input, int dim) {
    int batch_size, dim1, dim2;
    if (dim == 1) {
        batch_size = input.size(0);
        dim1 = input.size(1);
        dim2 = input.size(2);
    } else {
        // Handle other dimensions if needed
        AT_ASSERT(false, "Currently only supports reduction over dim 1");
    }

    auto output = torch::empty({batch_size, dim2}, input.options());

    int blocks = (batch_size * dim2 + 255) / 256;
    int threads = 256;

    min_kernel<float><<<blocks, threads>>>(
        input.packed_accessor<float, 2, torch::RestrictPtrTraits>(),
        output.packed_accessor<float, 1, torch::RestrictPtrTraits>(),
        dim1, dim2, batch_size * dim2, dim == 1 ? dim1 : dim2);

    return output;
}
"""

min_reduction_cpp_source = """
torch::Tensor min_reduction_cuda(torch::Tensor input, int dim);
"""

min_reduction = load_inline(
    name="min_reduction",
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_source,
    functions=["min_reduction_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.min_reduction = min_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_reduction.min_reduction_cuda(x, self.dim)