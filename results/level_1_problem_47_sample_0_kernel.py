import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sum_reduction_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename scalar_t>
__global__ void sum_reduction_kernel(const scalar_t* __restrict__ input,
                                    scalar_t* __restrict__ output,
                                    int outer_dim,
                                    int reduce_dim,
                                    int inner_dim) {
    extern __shared__ scalar_t shared_buf[];
    int tid = threadIdx.x;
    int block_idx = blockIdx.x;

    // Compute output index
    int out_idx = block_idx * inner_dim + tid;

    // Each block handles a "slice" along the reduce dimension
    scalar_t local_sum = 0;
    for (int r = 0; r < reduce_dim; ++r) {
        int in_idx = block_idx * reduce_dim * inner_dim + r * inner_dim + tid;
        local_sum += input[in_idx];
    }

    // Synchronous sum reduction in shared memory
    shared_buf[tid] = local_sum;
    __syncthreads();

    for (int s = reduce_dim >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            shared_buf[tid] += shared_buf[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[block_idx * inner_dim] = shared_buf[0];
    }
}

std::vector<torch::Tensor> sum_reduction_cuda(torch::Tensor input, int dim) {
    const int64_t* input_sizes = input.sizes().data();
    int ndims = input.dim();

    // Compute the size along the reduction dimension
    int reduce_size = input.size(dim);

    // Compute the size of the output tensor (keeping dim as size 1)
    auto output_sizes = input.sizes().vec();
    output_sizes[dim] = 1;

    // Calculate outer and inner dimensions
    int outer_dim = 1;
    for (int i = 0; i < dim; ++i) {
        outer_dim *= input.size(i);
    }
    int inner_dim = 1;
    for (int i = dim + 1; i < ndims; ++i) {
        inner_dim *= input.size(i);
    }

    // Number of blocks required = outer_dim * inner_dim / reduce_size ??
    int num_blocks = outer_dim * inner_dim;
    int block_size = 256;

    // Output tensor allocation
    auto output = torch::empty(output_sizes, input.options());

    // Shared memory size = block_size * sizeof(float)
    dim3 threads(block_size);
    dim3 blocks(num_blocks);

    sum_reduction_kernel<float><<<blocks, threads, block_size * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        outer_dim,
        reduce_size,
        inner_dim
    );

    cudaDeviceSynchronize();
    return {output};
}
"""

sum_reduction_cuda_header = """
std::vector<torch::Tensor> sum_reduction_cuda(torch::Tensor input, int dim);
"""

sum_reduction = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_cuda_header,
    cuda_sources=sum_reduction_cuda_source,
    functions="sum_reduction_cuda",
    verbose=True,
    extra_cflags=["-DWITH_CUDA", "-x c++", "-std=c++14"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduction = sum_reduction

    def forward(self, x):
        return self.sum_reduction.sum_reduction_cuda(x, self.dim)[0]