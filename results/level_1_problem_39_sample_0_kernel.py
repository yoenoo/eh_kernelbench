import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

__global__ void l2_norm_kernel(const float* x, float* out, float* norms, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    float sum = 0.0;
    for (int i = batch_idx * dim; i < (batch_idx + 1) * dim; i += stride) {
        float val = x[i];
        sum += val * val;
    }

    __shared__ float shared_sum[1024];
    int tid = threadIdx.x;
    shared_sum[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        norms[batch_idx] = sqrt(shared_sum[0]);
    }
    __syncthreads();

    for (int i = batch_idx; i < dim; i += stride) {
        int idx = batch_idx * dim + i;
        out[idx] = x[idx] / norms[batch_idx];
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int dim = x.size(1);

    auto out = torch::empty_like(x);
    auto norms = torch::empty({batch_size}, x.options());

    const int block_size = 256;
    const int num_blocks = batch_size; // Each block handles a batch element

    l2_norm_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        norms.data_ptr<float>(),
        batch_size,
        dim
    );

    return out;
}
"""

l2_norm_cpp_source = "torch::Tensor l2_norm_cuda(torch::Tensor x);"

l2norm = load_inline(
    name="l2_norm",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_norm = l2norm

    def forward(self, x):
        return self.l2_norm.l2_norm_cuda(x)