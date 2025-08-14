import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void l1_norm_forward_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Compute the sum of absolute values along dim=1
    scalar_t sum = 0;
    for (int i = tid; i < dim; i += blockDim.x) {
        scalar_t val = abs(x[batch_idx * dim + i]);
        sum += val;
    }

    // Block reduction to get the total sum for each batch element
    __shared__ scalar_t shared_sum[256];
    shared_sum[tid] = sum;
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
    }

    if (tid == 0) {
        scalar_t inv_sum = 1.0 / static_cast<scalar_t>(dim);  // L1 normalization divides by sum of absolute values
        shared_sum[0] = inv_sum / shared_sum[0];
    }
    __syncthreads();

    // Apply normalization
    scalar_t inv_sum = shared_sum[0];
    for (int i = tid; i < dim; i += blockDim.x) {
        y[batch_idx * dim + i] = x[batch_idx * dim + i] * inv_sum;
    }
}

std::tuple<torch::Tensor, torch::Tensor> l1_norm_cuda_forward(torch::Tensor x) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    auto y = torch::empty_like(x);
    auto sum_buf = torch::empty(batch_size, 1, x.options());

    const int block_size = 256;
    const dim3 grid(batch_size);
    const dim3 block(block_size);

    // Launch kernel with appropriate type
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "l1_norm_forward", ([&] {
        l1_norm_forward_kernel<scalar_t><<<grid, block>>>(
            x.data<scalar_t>(), y.data<scalar_t>(), batch_size, dim);
    }));

    return std::make_tuple(y, sum_buf); // dummy sum_buf for compatibility
}
"""

cpp_source = """
std::tuple<torch::Tensor, torch::Tensor> l1_norm_cuda_forward(torch::Tensor x);
"""

l1_norm = load_inline(
    name="l1_norm",
    cpp_sources=cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_norm_cuda_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l1_norm = l1_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.l1_norm.l1_norm_cuda_forward(x.cuda())
        return y