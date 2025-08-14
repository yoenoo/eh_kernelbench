import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L2 normalization
l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void l2_norm_kernel(const float* x, float* out, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int element_idx = threadIdx.x;

    if (element_idx >= dim) return;

    float sum = 0.0;
    for (int i = 0; i < dim; ++i) {
        float val = x[batch_idx * dim + i];
        sum += val * val;
    }
    float norm = rsqrtf(sum + 1e-8); // Add small epsilon for stability

    for (int i = element_idx; i < dim; i += blockDim.x) {
        int idx = batch_idx * dim + i;
        out[idx] = x[idx] * norm;
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int dim = x.size(1);

    auto out = torch::empty_like(x);

    const int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(dim < 256 ? dim : 256); // Adjust block size if needed

    l2_norm_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim);

    return out;
}
"""

l2_norm_cpp_source = (
    "torch::Tensor l2_norm_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for L2 normalization
l2_norm = load_inline(
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
        self.l2_norm = l2_norm

    def forward(self, x):
        return self.l2_norm.l2_norm_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []