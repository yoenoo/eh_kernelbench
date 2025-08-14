import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for KL divergence
kl_div_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void kl_div_kernel(const scalar_t* __restrict__ p, 
                             const scalar_t* __restrict__ q,
                             scalar_t* output,
                             int batch_size,
                             int dim_size) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ scalar_t shared_sum[32]; // Assuming max block size 1024 with 32 warps, or adjust dynamically
    
    shared_sum[tid] = 0;
    
    for (int i = tid; i < dim_size; i += blockDim.x) {
        scalar_t p_val = p[batch * dim_size + i];
        scalar_t q_val = q[batch * dim_size + i];
        if (p_val > 0 && q_val > 0) { // NaN checks
            shared_sum[tid] += p_val * (log(p_val) - log(q_val));
        }
    }
    
    __syncthreads();
    
    // Reduce sum across the block
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, shared_sum[0]);
    }
}

torch::Tensor kl_div_cuda(torch::Tensor p, torch::Tensor q) {
    // Ensure p and q have same shape
    const auto batch_size = p.size(0);
    const auto dim_size = p.size(1);
    
    torch::Tensor output = torch::zeros(1, device p.device(), dtype p.dtype());
    
    const int block_size = 256; // Tune based on occupancy
    const int num_blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "kl_div_cuda", ([&] {
        kl_div_kernel<scalar_t><<<num_blocks, block_size>>>(
            p.data<scalar_t>(),
            q.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            dim_size
        );
    }));
    
    // Compute the mean over the batch and divide by batch_size (since atomicAdd sums all terms)
    output = output / (batch_size * dim_size);
    return output;
}
"""

cpp_source = """
torch::Tensor kl_div_cuda(torch::Tensor p, torch::Tensor q);
"""

# Compile the CUDA kernel
kl_div_extension = load_inline(
    name="kl_div",
    cpp_sources=cpp_source,
    cuda_sources=kl_div_kernel,
    functions=["kl_div_cuda"],
    verbose=False,
    extra_cflags=["-g", "-O3"],
    extra_cuda_cflags=["-g", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.kl_div = kl_div_extension

    def forward(self, predictions, targets):
        return self.kl_div.kl_div_cuda(predictions, targets)