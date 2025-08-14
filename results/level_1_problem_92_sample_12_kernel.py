import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Define and load the custom CUDA kernel
        cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

template <typename scalar_t>
__global__ void exclusive_cumsum_kernel(scalar_t* out, const scalar_t* in, int dim_size, int outer_dim_size, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Compute the position along the cumulative dimension
    int pos_in_dim = idx % dim_size;
    if (pos_in_dim == 0) {
        out[idx] = 0;
    } else {
        int prev_idx = idx - 1;
        if (((idx / dim_size) * dim_size) <= prev_idx) {
            out[idx] = out[prev_idx];
        } else {
            out[idx] = 0;
        }
        out[idx] += in[prev_idx];
    }
}

template <typename scalar_t>
void exclusive_cumsum_cuda(torch::Tensor out, torch::Tensor in, int dim) {
    const auto total_elements = in.numel();
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    auto stream = at::cuda::current_stream().stream();
    const auto dim_size = in.size(dim);
    const auto outer_dim_size = in.numel() / dim_size;

    exclusive_cumsum_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        out.data_ptr<scalar_t>(), in.data_ptr<scalar_t>(), dim_size, outer_dim_size, total_elements);
}

torch::Tensor exclusive_cumsum(torch::Tensor in, int64_t dim) {
    auto out = torch::empty_like(in);
    AT_DISPATCH_FLOATING_TYPES(in.scalar_type(), "exclusive_cumsum_cuda", ([&] {
        exclusive_cumsum_cuda<scalar_t>(out, in, dim);
    }));
    return out;
}
"""

        cumsum_cuda = load_inline(
            name="exclusive_cumsum",
            cpp_sources="",
            cuda_sources=cumsum_source,
            functions=["exclusive_cumsum"],
            verbose=False
        )

        self.exclusive_cumsum = cumsum_cuda.exclusive_cumsum

    def forward(self, x):
        # Apply the exclusive cumulative sum using the custom kernel
        return self.exclusive_cumsum(x, self.dim)

def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    dim = 1
    return [dim]