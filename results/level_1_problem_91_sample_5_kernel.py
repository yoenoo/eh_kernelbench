import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for reverse cumulative sum
reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void reverse_cumsum_kernel(const scalar_t* x, scalar_t* out, int dim_size, int total_elements, int dim_stride, int dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_elements) return;

    int batch_idx = index / dim_size;
    int dim_offset = index % dim_size;

    // Compute original and reversed indices
    int original_pos = batch_idx * dim_size + (dim_size - 1 - dim_offset);
    int reversed_pos = batch_idx * dim_size + dim_offset;

    // Flip the dimension
    scalar_t* current_out = out + reversed_pos;
    const scalar_t* current_x = x + original_pos;

    // Perform reverse cumsum
    *current_out = *current_x;
    for (int i = 1; i <= dim_offset; ++i) {
        int prev_reversed_pos = reversed_pos - i * dim_stride;
        *current_out += x[prev_reversed_pos];
    }
}

std::vector<int64_t> get_launch_params(torch::Tensor x, int dim) {
    auto sizes = x.sizes().vec();
    auto total_elements = x.numel();
    int dim_size = sizes[dim];
    int dim_stride = 1;
    for (size_t i = dim + 1; i < sizes.size(); ++i) {
        dim_stride *= sizes[i];
    }

    return {total_elements, dim_size, dim_stride};
}

torch::Tensor reverse_cumsum_cuda(torch::Tensor x, int dim) {
    auto options = x.options();
    auto out = torch::empty_like(x);
    auto params = get_launch_params(x, dim);
    int total_elements = params[0];
    int dim_size = params[1];
    int dim_stride = params[2];

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    AT_DISPATCH_ALL_TYPES(x.type(), "reverse_cumsum_cuda", ([&] {
        reverse_cumsum_kernel<scalar_t><<<num_blocks, block_size>>>(
            x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), dim_size, total_elements, dim_stride, dim);
    }));

    return out;
}
"""

reverse_cumsum_header = """
std::vector<int64_t> get_launch_params(torch::Tensor x, int dim);
torch::Tensor reverse_cumsum_cuda(torch::Tensor x, int dim);
"""

reverse_cumsum = load_inline(
    name="reverse_cumsum",
    cpp_sources=reverse_cumsum_header,
    cuda_sources=reverse_cumsum_source,
    functions=["reverse_cumsum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return reverse_cumsum.reverse_cumsum_cuda(x, self.dim)