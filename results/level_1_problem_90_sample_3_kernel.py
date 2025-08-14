import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cumprod_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void cumulative_product_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t dim_size,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_stride) {
    extern __shared__ scalar_t shared_data[];

    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    // Each block processes a single slice along the dimension
    for (int i = block_idx * inner_size + thread_idx; i < outer_size * inner_size; i += gridDim.x * inner_size) {
        int outer = i / inner_size;
        int pos = outer * dim_stride + thread_idx;
        int output_pos = outer * dim_stride + thread_idx;

        if (thread_idx == 0) {
            shared_data[thread_idx] = input[pos];
        } else {
            shared_data[thread_idx] = input[pos] * shared_data[thread_idx - 1];
        }
        __syncthreads();

        output[output_pos] = shared_data[thread_idx];
    }
}

torch::Tensor cumulative_product_cuda(torch::Tensor input, int64_t dim) {
    const int64_t dims[] = input.sizes().vec();
    int64_t dim_size = input.size(dim);
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= dims[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= dims[i];
    }
    int64_t dim_stride = input.stride(dim);

    auto output = torch::empty_like(input);

    const int block_size = 256;
    dim3 block(block_size);
    dim3 grid(std::min(outer_size * inner_size, 65535LL));

    // Shared memory size is block_size * sizeof(scalar_t)
    size_t shared_mem_size = block_size * sizeof(float);

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "cumulative_product_cuda", ([&] {
        cumulative_product_kernel<scalar_t><<<grid, block, shared_mem_size, torch::cuda::current_stream()>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            outer_size,
            inner_size,
            dim_stride);
    }));

    return output;
}
"""

cumprod_cpp_source = "torch::Tensor cumulative_product_cuda(torch::Tensor input, int64_t dim);"

cumprod_extension = load_inline(
    name="cumprod_extension",
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_source,
    functions=["cumulative_product_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.cumulative_product_cuda = cumprod_extension

    def forward(self, x):
        return self.cumulative_product_cuda.cumulative_product_cuda(x, self.dim)