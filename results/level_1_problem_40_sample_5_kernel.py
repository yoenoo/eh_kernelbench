import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for LayerNorm
layer_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

template <typename T>
__global__ void layer_norm_kernel(const T* x, T* y, T* mean, T* invstd, T eps, int M, int N) {
    extern __shared__ char shmem[];
    T* shared_data = reinterpret_cast<T*>(shmem);
    T* thread_data = shared_data + threadIdx.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += gridDim.x * blockDim.x) {
        int row = i / N;
        int col = i % N;
        T val = x[i];
        thread_data[threadIdx.x] = val;
        __syncthreads();

        T sum = 0;
        #pragma unroll
        for (int t = 0; t < blockDim.x; ++t) {
            sum += shared_data[t];
        }
        sum = sum / N;
        mean[row] = sum;

        T var = 0;
        #pragma unroll
        for (int t = 0; t < blockDim.x; ++t) {
            var += (shared_data[t] - sum) * (shared_data[t] - sum);
        }
        var = var / N + eps;
        invstd[row] = 1.0 / sqrt(var);

        y[i] = (val - sum) * invstd[row];
        __syncthreads();
    }
}

torch::Tensor layer_norm_cuda(torch::Tensor x, float eps) {
    const int M = x.size(0) * x.size(1) * x.size(2) * x.size(3);
    const int N = x.size(-1); // last dimension as features
    const int threads = 256;
    const int blocks = (M + threads - 1) / threads;

    auto options = x.options();
    auto y = torch::empty_like(x);
    auto mean = torch::empty({M / N}, options);
    auto invstd = torch::empty({M / N}, options);

    layer_norm_kernel<float><<<blocks, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(), 
        y.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        eps,
        M,
        N
    );

    return y;
}
"""

# Compile the CUDA kernel
layer_norm_cpp = load_inline(
    name="layer_norm_cuda",
    cpp_sources="",
    cuda_sources=layer_norm_source,
    functions=["layer_norm_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = 1e-5  # Using default PyTorch epsilon
        self.layer_norm = layer_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm.layer_norm_cuda(x, self.eps)

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]