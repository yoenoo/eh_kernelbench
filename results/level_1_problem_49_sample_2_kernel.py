import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Max reduction
max_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void max_reduction_kernel(const scalar_t* input, scalar_t* output, const int batch_size, const int dim1, const int dim2, const int dim_to_reduce) {
    int batch_idx = blockIdx.x;
    int out_dim1_idx = blockIdx.y;
    int out_dim2_idx = threadIdx.x;

    int input_dim;
    int out_dim;
    if (dim_to_reduce == 0) {
        input_dim = batch_size;
        out_dim = 1;
    } else if (dim_to_reduce == 1) {
        input_dim = dim1;
        out_dim = dim2;
    } else {
        input_dim = dim2;
        out_dim = dim1;
    }

    int output_offset = (batch_idx * out_dim * (dim_to_reduce == 1 ? dim2 : dim1)) + (out_dim1_idx * out_dim);
    output_offset += out_dim2_idx;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    for (int i = 0; i < input_dim; ++i) {
        int input_offset = 0;
        if (dim_to_reduce == 0) {
            input_offset = (i * dim1 * dim2) + (out_dim1_idx * dim2) + out_dim2_idx;
        } else if (dim_to_reduce == 1) {
            input_offset = (batch_idx * dim1 * dim2) + (i * dim2) + out_dim2_idx;
        } else {
            input_offset = (batch_idx * dim1 * dim2) + (out_dim1_idx * dim2) + i;
        }
        if (input[input_offset] > max_val) {
            max_val = input[input_offset];
        }
    }
    output[output_offset] = max_val;
}

std::vector<int64_t> get_output_shape(int dim_to_reduce, const torch::Tensor& input) {
    auto input_shape = input.sizes().vec();
    input_shape.erase(input_shape.begin() + dim_to_reduce);
    return input_shape;
}

torch::Tensor max_reduction_cuda(torch::Tensor input, int dim) {
    auto output_shape = get_output_shape(dim, input);
    auto batch_size = output_shape[0];
    auto dim1 = (dim == 1) ? output_shape[1] : input.size(1);
    auto dim2 = (dim == 2) ? output_shape[1] : input.size(2);
    auto output = torch::empty(output_shape, input.options());

    dim3 blocks(batch_size);
    dim3 threads(output_shape[1] * output_shape[2]);

    int shared_mem = 0;
    max_reduction_kernel<float><<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input.size(1),
        input.size(2),
        dim
    );

    return output;
}
"""

max_reduction_cpp_source = """
std::vector<int64_t> get_output_shape(int dim_to_reduce, const torch::Tensor& input);
torch::Tensor max_reduction_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code for max reduction
max_reduction = load_inline(
    name="max_reduction",
    cpp_sources=max_reduction_cpp_source,
    cuda_sources=max_reduction_source,
    functions=["max_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.max_reduction = max_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_reduction.max_reduction_cuda(x, self.dim)

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return []