import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom Instance Norm CUDA kernel implementation
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void instance_norm_forward_kernel(
    const scalar_t* __restrict__ input, 
    scalar_t* __restrict__ output,
    const int batch_size, 
    const int channels,
    const int spatial_dim,
    const float eps) {

    const int hws = spatial_dim;
    const int hwa = channels * hws;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        for (int c = 0; c < channels; ++c) {
            scalar_t mean = 0.0;
            for (int s = 0; s < hws; ++s) {
                mean += input[idx * hwa + c * hws + s];
            }
            mean /= hws;

            scalar_t var = 0.0;
            for (int s = 0; s < hws; ++s) {
                scalar_t val = input[idx * hwa + c * hws + s] - mean;
                var += val * val;
            }
            var /= hws;
            scalar_t inv_std = 1.0 / sqrt(var + eps);

            for (int s = 0; s < hws; ++s) {
                output[idx * hwa + c * hws + s] = 
                    (input[idx * hwa + c * hws + s] - mean) * inv_std;
            }
        }
    }
}

torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    float eps = 1e-5) {

    const auto batch_size = input.size(0);
    const int channels = input.size(1);
    const int spatial_dim = input.size(2) * input.size(3);
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "instance_norm_forward", ([&] {
        instance_norm_forward_kernel<scalar_t>
        <<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, channels, spatial_dim, eps);
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

# Inline compilation of CUDA kernel
instance_norm = load_inline(
    name='instance_norm',
    cpp_sources="",
    cuda_sources=instance_norm_source,
    functions=['instance_norm_forward_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.instance_norm = instance_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.instance_norm.instance_norm_forward_cuda(x, self.eps)