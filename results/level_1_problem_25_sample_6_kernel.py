import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

swish_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void swish_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        scalar_t sigmoid_x = 1.0 / (1.0 + exp(-x[idx]));
        y[idx] = x[idx] * sigmoid_x;
    }
}

int swish_cuda(torch::Tensor x, torch::Tensor y) {
    const int n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    // Select appropriate kernel based on data type
    AT_DISPATCH_FLOATING_TYPES(x.type(), "swish_cuda", ([&] {
        swish_kernel<scalar_t><<<blocks, threads>>>(x.data<scalar_t>(), y.data<scalar_t>(), n);
    }));
    
    return 1;
}
"""

swish_cuda_header = "int swish_cuda(torch::Tensor x, torch::Tensor y);"

# Compile the inline CUDA code
swish_cuda = load_inline(
    name='swish_cuda',
    cpp_sources=swish_cuda_header,
    cuda_sources=swish_cuda_source,
    functions=('swish_cuda',),
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.swish_op = swish_cuda

    def forward(self, x):
        y = torch.empty_like(x)
        self.swish_op.swish_cuda(x, y)
        return y