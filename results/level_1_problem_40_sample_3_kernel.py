import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom LayerNorm CUDA kernel
layer_norm_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

template <typename scalar_t>
__global__ void layer_norm_fwd_kernel(
    const scalar_t* __restrict__ x, scalar_t* y, 
    const scalar_t* __restrict__ weight, const scalar_t* __restrict__ bias,
    const int N, const int C, const int H,
    float eps) {

    extern __shared__ char scratch[];
    float* shared_data = reinterpret_cast<float*>(scratch);

    int n = blockIdx.x;
    int c = threadIdx.x;
    int h = blockIdx.y;

    const int elem_size = C * H;
    const int offset = n * elem_size + c * H + h;

    // Load data into shared memory
    __shared__ float shared_x[C];
    shared_x[c] = x[offset];
    __syncthreads();

    // Compute mean
    float mean = 0;
    for (int i = 0; i < C; ++i) {
        mean += shared_x[i];
    }
    mean /= C;

    // Compute variance
    float var = 0;
    for (int i = 0; i < C; ++i) {
        var += (shared_x[i] - mean) * (shared_x[i] - mean);
    }
    var /= C;
    var = 1.0f / sqrt(var + eps);

    __syncthreads();

    // Normalize and apply affine
    scalar_t inv_var = static_cast<scalar_t>(var);
    scalar_t normed = static_cast<scalar_t>(shared_x[c] - mean) * inv_var;

    y[offset] = normed * weight[c] + bias[c];
}

at::Tensor layer_norm_forward_cuda(
    at::Tensor x, at::Tensor weight, at::Tensor bias,
    float eps) {

    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2); // Assuming 2D spatial dimensions for simplicity

    auto y = at::empty_like(x);
    
    dim3 blocks(B, H);
    dim3 threads(C);

    layer_norm_fwd_kernel<float><<<blocks, threads, C * sizeof(float)>>>(
        x.data_ptr<float>(), y.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        B, C, H, eps);

    return y;
}
"""

class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape[0]))
        self.bias = nn.Parameter(torch.zeros(normalized_shape[0]))
        self.eps = 1e-5

        # Load custom CUDA kernel
        self.layer_norm = load_inline(
            name='layer_norm',
            cuda Sources=layer_norm_source,
            functions=['layer_norm_forward_cuda'],
            verbose=True
        )

    def forward(self, x):
        # Ensure input is on GPU
        x = x.cuda()
        weight = self.weight.cuda()
        bias = self.bias.cuda()

        # Reshape for kernel compatibility (assumes 3D tensor: B, C, H)
        original_shape = x.shape
        x_reshaped = x.view(x.size(0), self.normalized_shape[0], -1)

        # Execute custom kernel
        y = self.layer_norm.layer_norm_forward_cuda(x_reshaped, weight, bias, self.eps)

        # Restore original shape
        return y.view(original_shape)

# Adjust get_inputs() and get_init_inputs() if needed (remains same as original)