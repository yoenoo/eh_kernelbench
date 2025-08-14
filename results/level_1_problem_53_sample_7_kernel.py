import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for min reduction along a specific dimension
min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename scalar_t>
__global__ void min_reduction_kernel(const scalar_t* input, scalar_t* output,
                                    int dim_size, int outer_size, int inner_size,
                                    int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size * inner_size) {
        return;
    }

    int outer = idx / inner_size;
    int inner = idx % inner_size;

    scalar_t min_val = std::numeric_limits<scalar_t>::max();
    int offset = outer * dim_size * inner_size + inner;

    for (int d = 0; d < dim_size; ++d) {
        scalar_t val = input[offset + d * inner_size];
        if (val < min_val) {
            min_val = val;
        }
    }

    output[outer * inner_size + inner] = min_val;
}

std::vector<torch::Tensor> min_reduction_cuda(torch::Tensor input, int dim) {
    int dims[] = {input.size(0), input.size(1), input.size(2)};
    int ndim = input.dim();
    int dim_size = input.size(dim);
    int outer_dim = 1;
    int inner_dim = 1;

    // Compute outer and inner dimensions based on the reduction dimension
    for (int i = 0; i < ndim; ++i) {
        if (i < dim) {
            outer_dim *= dims[i];
        } else if (i > dim) {
            inner_dim *= dims[i];
        }
    }

    int outer_size = outer_dim;
    int inner_size = inner_dim;
    int total_elements = outer_size * inner_size;

    auto output = torch::empty({outer_size, inner_size}, input.options());

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // Launch the CUDA kernel
    min_reduction_kernel<float><<<num_blocks, block_size>>>(
        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
        dim_size, outer_size, inner_size, dim);

    // Reshape output tensor to match the expected output shape
    std::vector<int64_t> output_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i != dim) {
            output_shape.push_back(dims[i]);
        }
    }

    return {output.reshape(output_shape), torch::Tensor()}; // Dummy index tensor
}

torch::Tensor min_reduction_forward(torch::Tensor input, int dim) {
    auto result = min_reduction_cuda(input, dim);
    return result[0]; // Return only the values, not the indices
}
"""

min_reduction_cpp_source = """
std::vector<torch::Tensor> min_reduction_cuda(torch::Tensor input, int dim);
torch::Tensor min_reduction_forward(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
min_reduction = load_inline(
    name="min_reduction",
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_source,
    functions=["min_reduction_forward"],
    verbose=True,
    extra_cflags=["-DVERSION_GE_1_5"],
    extra_cuda_cflags=["--expt-relaxed-constexpr"],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.min_reduction = min_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_reduction.min_reduction_forward(x, self.dim)