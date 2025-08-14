import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

template <typename scalar_t>
__global__ void compute_frobenius_norm_kernel(const scalar_t* data, scalar_t* norm, int64_t size) {
    extern __shared__ scalar_t shared_buf[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    scalar_t sum = 0;
    if (idx < size) {
        sum = data[idx] * data[idx];
    }
    __shared__ scalar_t partial_sum;
    partial_sum = 0;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        partial_sum += data[i] * data[i];
    }
    shared_buf[threadIdx.x] = partial_sum;
    __syncthreads();
    
    int s = blockDim.x;
    while (s > 1) {
        int mid = s / 2;
        if (threadIdx.x < mid) {
            shared_buf[threadIdx.x] += shared_buf[threadIdx.x + mid];
        }
        __syncthreads();
        s = mid;
    }
    if (threadIdx.x == 0) {
        *norm = sqrt(static_cast<double>(shared_buf[0]));
    }
}

template <typename scalar_t>
__global__ void divide_by_norm_kernel(const scalar_t* data, scalar_t* output, scalar_t norm, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = static_cast<scalar_t>(data[idx] / norm);
    }
}

torch::Tensor compute_frobenius_norm_cuda(const torch::Tensor& data) {
    int block_size = 1024;
    int grid_size = (data.numel() + block_size - 1) / block_size;
    torch::Tensor norm_tensor = torch::empty(1, data.options());
    compute_frobenius_norm_kernel<float><<<grid_size, block_size, block_size * sizeof(float), cudaStreamDefault>>>(
        data.data_ptr<float>(), norm_tensor.data_ptr<float>(), data.numel()
    );
    cudaDeviceSynchronize();
    return norm_tensor;
}

torch::Tensor frobenius_normalize_cuda(const torch::Tensor& x) {
    const auto x_size = x.sizes();
    auto output = torch::empty_like(x);
    auto norm = compute_frobenius_norm_cuda(x);
    
    const int block_size = 256;
    const int grid_size = (x.numel() + block_size - 1) / block_size;
    divide_by_norm_kernel<float><<<grid_size, block_size, 0, cudaStreamDefault>>>(
        x.data_ptr<float>(), 
        output.data_ptr<float>(), 
        norm.item<float>(), 
        x.numel()
    );
    cudaDeviceSynchronize();
    return output;
}
"""

frobenius_norm_header = """
torch::Tensor frobenius_normalize_cuda(const torch::Tensor& x);
"""

frobenius_norm_cuda = load_inline(
    name="frobenius_norm_cuda",
    cpp_sources=frobenius_norm_header,
    cuda_sources=frobenius_norm_source,
    functions=["frobenius_normalize_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.frobenius_normalize = frobenius_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frobenius_normalize.frobenius_normalize_cuda(x)