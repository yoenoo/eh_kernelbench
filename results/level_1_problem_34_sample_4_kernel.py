import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Instance Normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void instance_norm_forward_kernel(const torch::PackedTensorAccessor<scalar_t,4> input,
                                             torch::PackedTensorAccessor<scalar_t,4> output,
                                             scalar_t eps,
                                             int batch_size,
                                             int num_features,
                                             int height,
                                             int width) {
    int N = blockIdx.x * blockDim.x + threadIdx.x;
    if (N >= batch_size) return;

    int C = blockIdx.y * blockDim.y + threadIdx.y;
    if (C >= num_features) return;

    // Compute mean and variance for each (N, C) channel
    scalar_t mean = 0.0;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            mean += input[N][C][h][w];
        }
    }
    mean /= (height * width);

    scalar_t var = 0.0;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            var += (input[N][C][h][w] - mean) * (input[N][C][h][w] - mean);
        }
    }
    var /= (height * width);
    scalar_t std = sqrt(var + eps);

    // Normalize and write output
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            output[N][C][h][w] = (input[N][C][h][w] - mean) / std;
        }
    }
}

torch::Tensor instance_norm_forward_cuda(torch::Tensor input, float eps) {
    const auto batch_size = input.size(0);
    const auto num_features = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);

    auto output = torch::empty_like(input);

    dim3 threads(1, 1); // 1D thread block for N and C
    dim3 blocks(batch_size, num_features);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "instance_norm_forward", ([&] {
        instance_norm_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            eps,
            batch_size,
            num_features,
            height,
            width);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

instance_norm_cpp_source = (
    "torch::Tensor instance_norm_forward_cuda(torch::Tensor input, float eps);"
)

# Compile the inline CUDA code
instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.eps = 1e-5  # Use same epsilon as PyTorch's default
        self.instance_norm = instance_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.instance_norm.instance_norm_forward_cuda(x, self.eps)