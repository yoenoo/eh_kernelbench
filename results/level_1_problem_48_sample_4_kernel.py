import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void mean_reduction_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, int dim_size, int outer_size, int inner_size, int reduce_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= outer_size * inner_size) {
        return;
    }

    scalar_t sum = 0;
    int base_offset = index % inner_size;
    for (int d = 0; d < reduce_size; d++) {
        int offset = (index / inner_size) * dim_size * inner_size + (d * inner_size + base_offset);
        sum += input[offset];
    }
    output[index] = sum / reduce_size;
}

std::vector<int64_t> get_output_shape(int64_t dim, const at::Tensor& input) {
    auto input_shape = input.sizes().vec();
    input_shape.erase(input_shape.begin() + dim);
    return input_shape;
}

at::Tensor mean_reduction_cuda(const at::Tensor& input, int64_t dim) {
    const int threads_per_block = 256;
    auto input_shape = input.sizes();
    int64_t input_dim = input.dim();
    int64_t reduce_size = input.size(dim);
    dim = dim < 0 ? input_dim + dim : dim;

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input_dim; i++) {
        inner_size *= input.size(i);
    }
    int64_t output_elements = outer_size * inner_size;

    at::Tensor output = at::empty({output_elements}, input.options());
    const int block_size = threads_per_block;
    const int num_blocks = (output_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_reduction_cuda", ([&] {
        mean_reduction_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            input.size(dim),
            outer_size,
            inner_size,
            reduce_size
        );
    }));

    output = output.view(get_output_shape(dim, input));
    return output;
}
"""

mean_reduction_cpp_source = """
at::Tensor mean_reduction_cuda(const at::Tensor &input, int64_t dim);
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
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mean_reduction = mean_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_reduction.mean_reduction_cuda(x, self.dim)