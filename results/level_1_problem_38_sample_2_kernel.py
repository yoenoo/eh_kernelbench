import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void l1_norm_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, const int batch_size, const int dim) {
    int batch_idx = blockIdx.x;
    int element_idx = threadIdx.x;
    
    __shared__ scalar_t sum[256]; // Assuming blockDim.x <= 256 for simplicity
    
    sum[element_idx] = 0;
    for (int i = element_idx; i < dim; i += blockDim.x) {
        sum[element_idx] += abs(x[batch_idx * dim + i]);
    }
    
    __syncwarp();
    
    // Parallel reduction to compute the sum for each batch
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (element_idx < s) {
            sum[element_idx] += sum[element_idx + s];
        }
        __syncwarp();
    }
    
    if (element_idx == 0) {
        scalar_t inv_sum = 1.0 / (sum[0] + 1e-12); // Avoid division by zero
        for (int i = 0; i < dim; ++i) {
            y[batch_idx * dim + i] = x[batch_idx * dim + i] * inv_sum;
        }
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x) {
    const auto batch_size = x.size(0);
    const auto dim = x.size(1);
    auto y = torch::empty_like(x);
    
    const int block_size = 256; // Threads per block should be <= dim and a power of 2 for optimal performance
    const dim3 blocks(batch_size);
    const dim3 threads(block_size);
    
    AT_DISPATCH_FLOATING_TYPES(x.type(), "l1_norm_cuda", ([&] {
        l1_norm_kernel<scalar_t><<<blocks, threads>>>(x.data<scalar_t>(), y.data<scalar_t>(), batch_size, dim);
    }));
    
    return y;
}
"""

cpp_source = """
torch::Tensor l1_norm_cuda(torch::Tensor x);
"""

l1_norm = load_inline(
    name="l1_norm",
    cpp_sources=cpp_source,
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

    def forward(self, x):
        return self.l1_norm.l1_norm_cuda(x)