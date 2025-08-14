import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l1_normalize_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void l1_normalize_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out, const int batch_size, const int dim) {
    int batch_idx = blockIdx.x;
    int element_idx = threadIdx.x;
    
    __shared__ scalar_t sum[256]; // assuming max threads per block is 256, but can be adjusted
    sum[element_idx] = 0;
    
    for (int i = element_idx; i < dim; i += blockDim.x) {
        sum[element_idx] += fabs(x[batch_idx * dim + i]);
    }
    
    __syncthreads();
    
    // Compute the sum using reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (element_idx < s) {
            sum[element_idx] += sum[element_idx + s];
        }
        __syncthreads();
    }
    
    if (element_idx == 0) {
        scalar_t inv_sum = 1.0 / sum[0];
        for (int i = 0; i < dim; i++) {
            out[batch_idx * dim + i] = x[batch_idx * dim + i] * inv_sum;
        }
    }
}

torch::Tensor l1_normalize_cuda(torch::Tensor x) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    
    // Set up blocks and threads
    const int block_size = 256;
    const int grid_size = batch_size;
    
    auto out = torch::empty_like(x);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "l1_normalize_cuda", ([&] {
        l1_normalize_kernel<scalar_t><<<grid_size, block_size>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            dim
        );
    }));
    
    cudaDeviceSynchronize();
    return out;
}
"""

l1_normalize_cpp_source = (
    "torch::Tensor l1_normalize_cuda(torch::Tensor x);"
)

l1_normalize = load_inline(
    name="l1_normalize",
    cpp_sources=l1_normalize_cpp_source,
    cuda_sources=l1_normalize_source,
    functions=["l1_normalize_cuda"],
    verbose=True,
    extra_cflags=["-D要不然annotation可能会有问题"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l1_normalize = l1_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_normalize.l1_normalize_cuda(x)