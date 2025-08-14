import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

layer_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

template <typename T>
__global__ void layer_norm_kernel(const T* x, T* y, T* mean, T* invstd, T eps, int M, int N) {
    extern __shared__ char smem[];
    T* shared = reinterpret_cast<T*>(smem);
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    T local_sum = 0;
    T local_sumsq = 0;
    
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
        T val = x[i];
        local_sum += val;
        local_sumsq += val * val;
    }
    
    __syncthreads();
    
    // Reduce local sums into shared memory
    shared[tid] = local_sum;
    __syncthreads();
    if (tid == 0) {
        T total = 0;
        T total_sq = 0;
        for (int i = 0; i < blockDim.x; i++) {
            total += shared[i];
            total_sq += shared[i + blockDim.x];
        }
        shared[0] = total;
        shared[blockDim.x] = total_sq;
    }
    __syncthreads();
    
    if (tid < blockDim.x) {
        T total = (tid == 0) ? shared[0] : 0;
        T total_sq = (tid == 0) ? shared[blockDim.x] : 0;
        
        T mean_val = total / N;
        T var = total_sq / N - mean_val * mean_val;
        T inv_std = 1.0 / sqrt(var + eps);
        
        if (tid == 0) {
            mean[blockIdx.x] = mean_val;
            invstd[blockIdx.x] = inv_std;
        }
        
        // Normalize and write output
        for (int i = tid; i < N; i += blockDim.x) {
            int idx = blockIdx.x * N + i;
            y[idx] = (x[idx] - mean_val) * inv_std;
        }
    }
}

torch::Tensor layer_norm_cuda(torch::Tensor x, float eps) {
    const int M = x.size(0);
    const int N = x.size(1);
    const int threads = 256;
    
    auto y = torch::empty_like(x);
    auto mean = torch::empty({M}, x.options());
    auto invstd = torch::empty({M}, x.options());
    
    layer_norm_kernel<float><<<M, threads, 2 * threads * sizeof(float)>>>(
        x.data_ptr<float>(), y.data_ptr<float>(),
        mean.data_ptr<float>(), invstd.data_ptr<float>(),
        eps, M, N);
    
    return y;
}
"""

layer_norm_cpp_source = (
    "torch::Tensor layer_norm_cuda(torch::Tensor x, float eps);"
)

layer_norm = load_inline(
    name="layer_norm_cuda",
    cpp_sources=layer_norm_cpp_source,
    cuda_sources=layer_norm_source,
    functions=["layer_norm_cuda"],
    verbose=True,
    extra_cflags=["-g", "-O3", "-D_FORCE_INLINES"],
    extra_cuda_cflags=["--expt-extended-lambda"]
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = 1e-5  # Default PyTorch epsilon value
        self.layer_norm = layer_norm
        
    def forward(self, x):
        # Reshape to 2D for kernel (batch * spatial dims, features)
        orig_shape = x.shape
        x = x.view(orig_shape[0], -1)
        out = self.layer_norm.layer_norm_cuda(x, self.eps)
        return out.view(orig_shape)