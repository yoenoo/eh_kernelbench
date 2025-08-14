import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for min reduction along a specific dimension
min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/macros/Macros.h>

template <typename scalar_t>
__global__ void min_reduction_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output,
                                    int64_t dim_size, int64_t outer_dim, int64_t inner_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_dim * inner_dim) return;

    int outer = idx / inner_dim;
    int inner = idx % inner_dim;
    scalar_t min_val = input[outer * dim_size * inner_dim + inner];
    for (int d = 0; d < dim_size; ++d) {
        int pos = outer * dim_size * inner_dim + d * inner_dim + inner;
        if (input[pos] < min_val) {
            min_val = input[pos];
        }
    }
    output[idx] = min_val;
}

torch::Tensor min_reduction_cuda(torch::Tensor input, int64_t dim) {
    int64_t ndim = input.dim();
    int64_t dim_size = input.size(dim);
    auto other_sizes = input.sizes().vec();
    other_sizes.erase(other_sizes.begin() + dim);

    auto output_size = torch::IntArrayRef(other_sizes);
    auto output = torch::empty(output_size, input.options());

    int64_t total_elements = output.numel();
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    int64_t outer_dim = 1;
    for (int i = 0; i < dim; ++i) {
        outer_dim *= input.size(i);
    }
    int64_t inner_dim = 1;
    for (int i = dim + 1; i < input.dim(); ++i) {
        inner_dim *= input.size(i);
    }

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "min_reduction_cuda", ([&] {
        min_reduction_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            dim_size,
            outer_dim,
            inner_dim
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

min_reduction_cpp_source = (
    "torch::Tensor min_reduction_cuda(torch::Tensor input, int64_t dim);"
)

# Compile the inline CUDA code for min reduction
min_reduction = load_inline(
    name="min_reduction",
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_source,
    functions=["min_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.min_reduction = min_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_reduction.min_reduction_cuda(x, self.dim)

def get_inputs():
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]