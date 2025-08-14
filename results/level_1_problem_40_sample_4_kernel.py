import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

epsilon = 1e-5

layer_norm_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void layer_norm_forward_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, scalar_t* __restrict__ mean, scalar_t* __restrict__ invvar, int N, int C) {
    extern __shared__ scalar_t shared[];
    scalar_t* mean_buf = shared;
    scalar_t* var_buf = shared + (blockDim.x);

    int n = blockIdx.x;
    int c = threadIdx.x;
    scalar_t val = x[n * C + c];
    
    scalar_t local_mean = val;
    scalar_t local_var = 0;
    
    for (int stride = C/2; stride > 0; stride >>=1) {
        __shared__ scalar_t tmp;
        if (c < stride) {
            tmp = local_mean + local_var;
            local_mean += val;
            local_var = tmp;
        }
        __syncthreads();
    }

    if (c == 0) {
        mean[n] = local_mean / C;
        var_buf[c] = local_mean * local_mean / C;
        for (int i =0; i < C; i++) {
            var_buf[c] += (x[n*C + i] - mean[n]) * (x[n*C + i] - mean[n]);
        }
        var_buf[c] /= C;
        invvar[n] = 1.0 / sqrt(var_buf[c] + 1e-5);
    }
    __syncthreads();

    y[n*C + c] = (val - mean[n]) * invvar[n];
}

torch::Tensor layer_norm_forward(torch::Tensor x) {
    const auto N = x.size(0);
    const auto C = x.size(1);

    auto y = torch::empty_like(x);
    auto mean = torch::empty({N}, x.options());
    auto invvar = torch::empty({N}, x.options());

    const int threads = C;
    const int blocks = N;

    layer_norm_forward_kernel<float><<<blocks, threads, 2*threads * sizeof(float)>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        mean.data_ptr<float>(),
        invvar.data_ptr<float>(),
        N, C
    );

    return y;
}
"""

layer_norm_forward_cpp = "torch::Tensor layer_norm_forward(torch::Tensor x);"

layer_norm = load_inline(
    name="layer_norm_cuda",
    cpp_sources=layer_norm_forward_cpp,
    cuda_sources=layer_norm_source,
    functions=["layer_norm_forward"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_ln = layer_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to 2D for simplification (batch, features, H, W) -> (batch, features*H*W)
        x_reshaped = x.view(x.size(0), -1)
        y = self.elementwise_ln.layer_norm_forward(x_reshaped)
        return y.view_as(x)