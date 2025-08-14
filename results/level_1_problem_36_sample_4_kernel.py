import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

rms_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void rms_norm_forward_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, const int batch_size, const int features, const int dim_total, const float eps, const float inv_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim_total) {
        scalar_t mean_sq = 0.0;
        # Compute mean of squares for the feature dimension
        for (int f = 0; f < features; ++f) {
            int feat_idx = idx + f * dim_total;
            scalar_t val = x[feat_idx];
            mean_sq += val * val * inv_features;
        }
        scalar_t rms = rsqrt(mean_sq + eps);
        for (int f = 0; f < features; ++f) {
            int feat_idx = idx + f * dim_total;
            y[feat_idx] = x[feat_idx] * rms;
        }
    }
}

torch::Tensor rms_norm_forward_cuda(torch::Tensor x, float eps, int features) {
    const int batch_size = x.size(0);
    const int dim_total = x.numel() / (batch_size * features);
    const int threads = 256;
    const int blocks = (batch_size * dim_total + threads - 1) / threads;
    const float inv_features = 1.0 / features;

    auto y = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream();

    rms_norm_forward_kernel<float><<<blocks, threads, 0, stream>>>(
        x.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        batch_size,
        features,
        dim_total,
        eps,
        inv_features
    );

    return y;
}
"""

rms_norm_cpp_source = """
torch::Tensor rms_norm_forward_cuda(torch::Tensor x, float eps, int features);
"""

rms_norm = load_inline(
    name="rms_norm",
    cpp_sources=[rms_norm_cpp_source],
    cuda_sources=[rms_norm_source],
    functions=["rms_norm_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_norm_forward = rms_norm

    def forward(self, x):
        # Calculate the RMS along the feature dimension using CUDA kernel
        return self.rms_norm_forward.rms_norm_forward_cuda(x, self.eps, self.num_features)

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features]