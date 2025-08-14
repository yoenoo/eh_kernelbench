import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for RMSNorm
rms_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void rms_norm_forward_kernel(const scalar_t* __restrict__ x,
                                       scalar_t* __restrict__ out,
                                       const int batch_size,
                                       const int num_features,
                                       const int dim,
                                       const float eps) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * dim) {
        return;
    }

    // Compute the feature dimension index
    int batch_dim = batch_size * num_features * dim;
    int d = idx % dim;
    int feature_slice = (idx / dim) % num_features;
    int batch = (idx / dim) / num_features;

    // Each thread computes x^2 for its position
    scalar_t x_squared = x[batch * num_features * dim + feature_slice * dim + d];
    x_squared *= x_squared;

    // Reduction to compute mean across features
    __shared__ float shared_data[1024]; // adjust based on block size
    int tid = threadIdx.x;
    shared_data[tid] = static_cast<float>(x_squared);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = shared_data[0] / num_features;
        float rms = rsqrt(mean + eps);
        for (int i = 0; i < dim; ++i) {
            int global_idx = batch * num_features * dim + feature_slice * dim + i;
            out[global_idx] = static_cast<scalar_t>(x[global_idx] * rms);
        }
    }
}

at::Tensor rms_norm_forward_cuda(at::Tensor x, float eps) {
    const int batch_size = x.size(0);
    const int num_features = x.size(1);
    const int dim = x.size(2) * x.size(3); // Assuming 4D tensor with last two dims flattened

    auto output = at::empty_like(x);

    const int threads = 1024;
    const int elements_per_feature = dim;
    const int blocks = (batch_size * elements_per_feature + threads - 1) / threads;

    auto stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel with appropriate template based on dtype
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rms_norm_forward_cuda", ([&] {
        rms_norm_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            num_features,
            dim,
            eps
        );
    }));

    return output;
}
"""

rms_norm_cpp_source = """
at::Tensor rms_norm_forward_cuda(at::Tensor x, float eps);
"""

# Compile the RMSNorm CUDA kernel
rms_norm = load_inline(
    name="rms_norm",
    cpp_sources=rms_norm_cpp_source,
    cuda_sources=rms_norm_source,
    functions=["rms_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_norm_forward = rms_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to 4D tensor if needed (assuming input is 4D as per get_inputs)
        return self.rms_norm_forward.rms_norm_forward_cuda(x, self.eps)

# Ensure get_inputs and get_init_inputs are included as per original
batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features]