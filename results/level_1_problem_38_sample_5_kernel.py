import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L1 normalization
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void l1_norm_kernel(const float* x_data, float* out_data, int batch_size, int dim, int total_elements) {
    extern __shared__ unsigned char smem[];
    float* sum_s = (float*)smem;

    int element_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (threadIdx.x == 0) {
        sum_s[0] = 0.0;
    }
    __syncthreads();

    if (element_idx < dim) {
        float val = abs(x_data[batch_idx * dim + element_idx]);
        atomicAdd(sum_s, val);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        sum_s[0] = sum_s[0] / dim;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < dim; idx += blockDim.x) {
        float x_val = x_data[batch_idx * dim + idx];
        out_data[batch_idx * dim + idx] = x_val / sum_s[0];
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int dim = x.size(1);
    int total_elements = batch_size * dim;

    auto out = torch::empty_like(x);

    dim3 block(256);
    dim3 grid(batch_size);

    // Total shared memory per block for storing partial sums (1 float per block)
    int smem_size = sizeof(float);

    l1_norm_kernel<<<grid, block, smem_size, torch::cuda::current_stream()>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        dim,
        total_elements
    );

    return out;
}
"""

# Compile the inline CUDA code for L1 normalization
l1_norm = load_inline(
    name="l1_norm",
    cpp_sources="",
    cuda_sources=l1_norm_source,
    functions=["l1_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l1_norm = l1_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_norm.l1_norm_cuda(x)