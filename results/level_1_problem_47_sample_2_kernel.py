import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for sum reduction
reduce_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void reduce_sum_kernel(const scalar_t* input, scalar_t* output, int dim_size, int outer_size, int inner_size, int reduce_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size * inner_size) return;

    int outer = idx / inner_size;
    int inner = idx % inner_size;
    scalar_t sum = 0;
    for (int d = 0; d < dim_size; ++d) {
        int input_idx = outer * dim_size * inner_size + d * inner_size + inner;
        sum += input[input_idx];
    }
    output[idx] = sum;
}

std::vector<int64_t> get_output_shape(int64_t dim, const at::Tensor& input) {
    auto shape = input.sizes().vec();
    shape[dim] = 1;
    return shape;
}

at::Tensor reduce_sum_cuda(at::Tensor input, int64_t dim) {
    auto shape = input.sizes();
    int ndim = input.dim();
    if (dim < 0) dim += ndim;

    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= shape[i];
    }
    int reduce_size = shape[dim];
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        inner_size *= shape[i];
    }

    auto output_shape = get_output_shape(dim, input);
    auto output = at::empty(output_shape, input.options());

    const int threads = 1024;
    int elements = outer_size * inner_size;
    int blocks = (elements + threads - 1) / threads;

    const int device = input.device().index();
    cudaSetDevice(device);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "reduce_sum_cuda", ([&] {
        reduce_sum_kernel<scalar_t><<<blocks, threads, 0, at::cuda::current_stream().stream()>>>(
            input.data<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            outer_size,
            inner_size,
            reduce_size);
    }));

    return output;
}
"""

reduce_sum_cpp_source = """
std::vector<int64_t> get_output_shape(int64_t dim, const at::Tensor& input);
at::Tensor reduce_sum_cuda(const at::Tensor& input, int64_t dim);
"""

# Compile the custom CUDA kernel
reduce_sum = load_inline(
    name="reduce_sum",
    cpp_sources=reduce_sum_cpp_source,
    cuda_sources=reduce_sum_source,
    functions=["reduce_sum_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduce_sum = reduce_sum

    def forward(self, x):
        return self.reduce_sum.reduce_sum_cuda(x, self.dim)