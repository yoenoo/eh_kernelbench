cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L1 normalization
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void l1_norm_kernel(const float* x, float* out, float* mean_abs, int batch_size, int dim) {
    // Compute the absolute values and accumulate per batch
    extern __shared__ volatile float shared[];
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    shared[tid] = 0.0;
    __syncthreads();

    for (int i = tid; i < dim; i += blockDim.x) {
        shared[tid] += fabs(x[batch_idx * dim + i]);
    }

    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        mean_abs[batch_idx] = shared[0] / dim;
    }
    __syncthreads();

    // Normalize
    if (tid < dim) {
        float value = x[batch_idx * dim + tid];
        out[batch_idx * dim + tid] = value / mean_abs[batch_idx];
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int dim = x.size(1);

    auto out = torch::empty_like(x);
    auto mean_abs = torch::empty(batch_size, x.options().dtype(torch::kFloat32));

    const int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);
    int shared_size = block_size * sizeof(float);

    l1_norm_kernel<<<grid, block, shared_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), mean_abs.data_ptr<float>(), batch_size, dim);

    return out;
}
"""

# Compile the inline CUDA code for L1 normalization
l1_norm = load_inline(
    name="l1_norm",
    cpp_sources="",
    cuda_sources=l1_norm_source,
    functions=["l1_norm_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l1_norm = l1_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_norm.l1_norm_cuda(x)

batch_size = 32768
dim = 65535

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []