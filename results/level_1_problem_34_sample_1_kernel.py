import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Instance Normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void instance_norm_forward_kernel(const torch::PackedTensorAccessor<scalar_t,4> input,
                                            torch::PackedTensorAccessor<scalar_t,4> output,
                                            torch::PackedTensorAccessor<scalar_t,1> weight,
                                            torch::PackedTensorAccessor<scalar_t,1> bias,
                                            float eps) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    const int nhw = H * W;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N * C) {
        int n = idx / C;
        int c = idx % C;

        scalar_t *input_nchw = input[n][c].data();
        scalar_t *output_nchw = output[n][c].data();

        scalar_t mean = 0;
        scalar_t var = 0;

        // Compute mean
        scalar_t sum = 0;
        for (int i = 0; i < nhw; i++) {
            sum += input_nchw[i];
        }
        mean = sum / nhw;

        // Compute variance
        scalar_t sq_sum = 0;
        for (int i = 0; i < nhw; i++) {
            scalar_t tmp = input_nchw[i] - mean;
            sq_sum += tmp * tmp;
        }
        var = sq_sum / nhw;

        scalar_t std = 1.0 / sqrt(var + eps);

        // Apply normalization and affine transform
        for (int i = 0; i < nhw; i++) {
            scalar_t val = (input_nchw[i] - mean) * std;
            if (weight.size(0) > 0) val *= weight[c];
            if (bias.size(0) > 0) val += bias[c];
            output_nchw[i] = val;
        }
    }
}

torch::Tensor instance_norm_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float eps) {
    const int threads = 256;
    const int N = input.size(0);
    const int C = input.size(1);
    const int size = N * C;
    const int blocks = (size + threads - 1) / threads;

    auto output = torch::empty_like(input);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "instance_norm_forward", ([&] {
        instance_norm_forward_kernel<scalar_t>
            <<<blocks, threads>>>(
                input.packed_accessor<scalar_t,4,torch::DefaultCUDAIndexing>(),
                output.packed_accessor<scalar_t,4,torch::DefaultCUDAIndexing>(),
                weight.packed_accessor<scalar_t,1,torch::DefaultCUDAIndexing>(),
                bias.packed_accessor<scalar_t,1,torch::DefaultCUDAIndexing>(),
                eps);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

instance_norm_cpp_source = """
torch::Tensor instance_norm_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float eps);
"""

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
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = 1e-5
        self.instance_norm = instance_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.instance_norm.instance_norm_forward_cuda(
            x.cuda(),
            self.weight.cuda(),
            self.bias.cuda(),
            self.eps,
        )