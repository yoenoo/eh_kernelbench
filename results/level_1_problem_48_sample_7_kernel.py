import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void mean_reduction_kernel(const scalar_t* input, scalar_t* output, int dim_size, int outer_size, int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size * inner_size) return;

    int outer = idx / inner_size;
    int inner = idx % inner_size;

    scalar_t sum = 0;
    for (int d = 0; d < dim_size; ++d) {
        int input_idx = outer * dim_size * inner_size + d * inner_size + inner;
        sum += input[input_idx];
    }
    output[idx] = sum / dim_size;
}

torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim) {
    int dims[] = {0, 0, 0};
    input.sizes().copy_to(dims);

    int64_t dim0 = dims[0];
    int64_t dim1 = dims[1];
    int64_t dim2 = dims[2];

    int outer_size, inner_size;
    if (dim == 0) {
        outer_size = 1;
        inner_size = dim1 * dim2;
    } else if (dim == 1) {
        outer_size = dim0;
        inner_size = dim2;
    } else {
        outer_size = dim0 * dim1;
        inner_size = 1;
    }

    int output_elements = outer_size * inner_size;
    auto output = torch::empty({outer_size * inner_size}, torch::device("cuda"));

    const int block_size = 256;
    const int grid_size = (output_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_reduction_cuda", ([&] {
        mean_reduction_kernel<scalar_t><<<grid_size, block_size>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            dim_size,
            outer_size,
            inner_size
        );
    }));

    return output.view({dim0, dim1, dim2}.sizes_without_dim(dim));
}
"""

mean_reduction_cpp_source = """
torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim);
"""

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
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mean_reduction = mean_reduction

    def forward(self, x):
        return self.mean_reduction.mean_reduction_cuda(x, self.dim)