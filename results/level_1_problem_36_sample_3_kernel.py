import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

rms_norm_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void rms_norm_kernel(scalar_t* x, scalar_t* out, const int batch_size, 
                               const int features, const int elements_per_sample,
                               const float eps) {

    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= batch_size) return;

    const int x_offset = sample_idx * features * elements_per_sample;
    scalar_t sum = 0;
    for (int f = 0; f < features; ++f) {
        for (int e = 0; e < elements_per_sample; ++e) {
            scalar_t val = x[x_offset + f * elements_per_sample + e];
            sum += val * val;
        }
    }

    scalar_t rms = sqrt((sum / features) + eps);
    rms = 1.0 / rms;

    for (int f = 0; f < features; ++f) {
        for (int e = 0; e < elements_per_sample; ++e) {
            out[x_offset + f * elements_per_sample + e] = 
                x[x_offset + f * elements_per_sample + e] * rms;
        }
    }
}

torch::Tensor rms_norm_cuda(torch::Tensor x, float eps) {
    const int batch_size = x.size(0);
    const int features = x.size(1);
    const int elements_per_sample = x.numel() / (batch_size * features);

    auto out = torch::empty_like(x);

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "rms_norm_cuda", ([&]{
        rms_norm_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            features,
            elements_per_sample,
            eps
        );
    }));

    return out;
}
"""

rms_norm_header = """
torch::Tensor rms_norm_cuda(torch::Tensor x, float eps);
"""

rms_norm_ops = load_inline(
    name='rms_norm_cuda',
    cpp_sources=rms_norm_header,
    cuda_sources=rms_norm_kernel,
    functions=['rms_norm_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_norm_cuda = rms_norm_ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rms_norm_cuda.rms_norm_cuda(x, self.eps)