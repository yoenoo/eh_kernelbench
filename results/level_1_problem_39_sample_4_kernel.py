import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l2norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void l2norm_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out, const int batch_size, const int dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ scalar_t squared_sum;

    if (tid == 0) {
        squared_sum = 0.0;
    }
    __syncthreads();

    // Compute the squared sum of each row
    for (int i = tid; i < dim; i += blockDim.x) {
        scalar_t val = x[batch_idx * dim + i];
        atomicAdd(&squared_sum, val * val);
    }
    __syncthreads();

    // Compute the norm for this row
    scalar_t norm = sqrt(squared_sum);
    __syncthreads();

    // Normalize each element in the row
    for (int i = tid; i < dim; i += blockDim.x) {
        int idx = batch_idx * dim + i;
        out[idx] = x[idx] / norm;
    }
}

torch::Tensor l2norm_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int dim = x.size(1);

    auto out = torch::empty_like(x);

    const int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);

    // Launch kernel
    l2norm_kernel<float><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim
    );

    return out;
}
"""

l2norm_cpp_source = "torch::Tensor l2norm_cuda(torch::Tensor x);"

# Compile the inline CUDA code
l2norm = load_inline(
    name="l2norm",
    cpp_sources=l2norm_cpp_source,
    cuda_sources=l2norm_source,
    functions=["l2norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l2norm_cuda = l2norm

    def forward(self, x):
        return self.l2norm_cuda.l2norm_cuda(x)

def get_inputs():
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []