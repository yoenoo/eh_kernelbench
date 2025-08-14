import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softplus_source = """
#include <torch/extension.h>
#include <math.h>
#include <cuda_runtime.h>

#define BETA 1.0f
#define THRESHOLD 20.0f

__global__ void softplus_kernel(const float* x, float* y, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xi = x[idx];
        if (xi > THRESHOLD) {
            y[idx] = xi;
        } else if (xi < -THRESHOLD) {
            y[idx] = expf(xi) / BETA;
        } else {
            y[idx] = (logf(1.0f + expf(xi * BETA)))/BETA;
        }
    }
}

torch::Tensor softplus_cuda(torch::Tensor x) {
    auto n = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    softplus_kernel<<<grid_size, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    cudaDeviceSynchronize();
    return y;
}
"""

cpp_source = "torch::Tensor softplus_cuda(torch::Tensor x);"

softplus_op = load_inline(
    name="softplus_op",
    cpp_sources=cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = softplus_op

    def forward(self, x):
        return self.softplus.softplus_cuda(x)