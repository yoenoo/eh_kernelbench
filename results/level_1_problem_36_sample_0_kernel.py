import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

rms_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename scalar_t>
__global__ void rms_norm_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, 
                               const int batch_size, const int features, const int dim1, const int dim2,
                               const float eps) {
    int pos_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos_idx >= batch_size * dim1 * dim2) return;

    int offset = pos_idx * features;
    scalar_t sum_sq = 0.0;
    for (int f = 0; f < features; ++f) {
        scalar_t val = x[offset + f];
        sum_sq += val * val;
    }
    sum_sq /= features;
    scalar_t inv_rms = 1.0 / sqrt(sum_sq + eps);
    for (int f = 0; f < features; ++f) {
        y[offset + f] = x[offset + f] * inv_rms;
    }
}

torch::Tensor rms_norm_cuda(torch::Tensor x, float eps) {
    const int batch_size = x.size(0);
    const int features = x.size(1);
    const int dim1 = x.size(2);
    const int dim2 = x.size(3);
    const int num_positions = batch_size * dim1 * dim2;

    auto y = torch::empty_like(x);
    int block_size = 256;
    int num_blocks = (num_positions + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<num_blocks, block_size>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            batch_size, features, dim1, dim2, eps);
    }));

    cudaDeviceSynchronize();
    return y;
}
"""

rms_norm_cpp_source = """
torch::Tensor rms_norm_cuda(torch::Tensor x, float eps);
"""

rms_norm = load_inline(
    name="rms_norm",
    cpp_sources=rms_norm_cpp_source,
    cuda_sources=rms_norm_source,
    functions=["rms_norm_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_norm_cuda = rms_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return self.rms_norm_cuda.rms_norm_cuda(x, self.eps)
        else:
            # Fall back to PyTorch implementation if not on CUDA
            rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
            return x / rms