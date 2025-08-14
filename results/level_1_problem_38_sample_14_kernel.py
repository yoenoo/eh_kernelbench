import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L1 normalization
l1_normalize_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void l1_normalize_kernel(const float* x, float* out, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int elem_idx = threadIdx.x;

    // Compute the absolute sum for each batch element
    __shared__ float sum[256]; // Assuming block size <= 256 for simplicity

    sum[elem_idx] = 0.0;

    for (int i = elem_idx; i < dim; i += blockDim.x) {
        float val = abs(x[batch_idx * dim + i]);
        sum[elem_idx] += val;
    }

    __syncthreads();

    // Reduce sum within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (i < s) {
            sum[elem_idx] += sum[elem_idx + s];
        }
        __syncthreads();
    }

    // Write the final sum for each batch element
    if (elem_idx == 0) {
        float total_sum = sum[0];
        // Avoid division by zero
        if (total_sum == 0.0) {
            total_sum = 1.0; // Or handle differently as needed
        }
        for (int i = 0; i < dim; ++i) {
            out[batch_idx * dim + i] = x[batch_idx * dim + i] / total_sum;
        }
    }
}

torch::Tensor l1_normalize_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int dim = x.size(1);

    auto out = torch::empty_like(x);

    dim3 block(min(256, dim)); // Adjust block size based on dimension
    dim3 grid(batch_size);

    l1_normalize_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        dim
    );

    return out;
}
"""

l1_normalize_cpp_source = (
    "torch::Tensor l1_normalize_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for L1 normalization
l1_normalize = load_inline(
    name="l1_normalize",
    cpp_sources=l1_normalize_cpp_source,
    cuda_sources=l1_normalize_source,
    functions=["l1_normalize_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_normalize = l1_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_normalize.l1_normalize_cuda(x)

def get_inputs():
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []