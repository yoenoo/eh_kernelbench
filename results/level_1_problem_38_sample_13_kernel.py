import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void l1_norm_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out, const int batch_size, const int dim) {
    int batch_idx = blockIdx.x;
    int elem_idx = threadIdx.x;

    __shared__ scalar_t sum[32];  // Shared memory for partial sums. Assuming dim <= 32768*2, but can be adjusted

    sum[threadIdx.x] = 0.0;

    // Compute the absolute sum for the current batch element
    for (int i = elem_idx; i < dim; i += blockDim.x) {
        sum[threadIdx.x] += std::abs(x[batch_idx * dim + i]);
    }

    __syncthreads();

    // Sum reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum[threadIdx.x] += sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        scalar_t inv_sum = 1.0 / sum[0];
        for (int i = 0; i < dim; ++i) {
            out[batch_idx * dim + i] = x[batch_idx * dim + i] * inv_sum;
        }
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);

    auto out = torch::empty_like(x);

    const int block_size = 256; // Tune this for optimal performance
    dim3 grid(batch_size);
    dim3 block(std::min(block_size, dim));

    AT_DISPATCH_FLOATING_TYPES(x.type(), "l1_norm_cuda", ([&] {
        l1_norm_kernel<scalar_t><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), batch_size, dim);
    }));

    return out;
}
"""

l1_norm_cpp_source = (
    "torch::Tensor l1_norm_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for L1 normalization
l1_norm = load_inline(
    name="l1_norm",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l1_norm = l1_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_norm.l1_norm_cuda(x)