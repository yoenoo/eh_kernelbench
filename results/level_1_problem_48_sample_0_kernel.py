import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void mean_reduction_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, 
                                      int dim_size, int outer_dim, int inner_dim, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_dim * inner_dim) {
        return;
    }

    scalar_t sum = 0;
    int outer = idx / inner_dim;
    int inner = idx % inner_dim;

    for (int d = 0; d < dim_size; ++d) {
        int input_idx = outer * dim_size * inner_dim + d * inner_dim + inner;
        sum += input[input_idx];
    }

    output[idx] = sum / dim_size;
}

torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim) {
    auto input_dims = input.sizes().vec();
    int outer_dim = 1;
    for (int i = 0; i < dim; ++i) {
        outer_dim *= input_dims[i];
    }
    int dim_size = input_dims[dim];
    int inner_dim = 1;
    for (int i = dim + 1; i < input_dims.size(); ++i) {
        inner_dim *= input_dims[i];
    }

    auto output_sizes = input_dims;
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    const int block_size = 256;
    const int num_elements = outer_dim * inner_dim;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "mean_reduction_kernel", ([&] {
        mean_reduction_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            outer_dim,
            inner_dim,
            dim);
    }));

    return output;
}
"""

mean_reduction_cpp_source = """
torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code for mean reduction
mean_reduction = load_inline(
    name="mean_reduction",
    cpp_sources=mean_reduction_cpp_source,
    cuda_sources=mean_reduction_source,
    functions=["mean_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.mean_reduction = mean_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_reduction.mean_reduction_cuda(x, self.dim)