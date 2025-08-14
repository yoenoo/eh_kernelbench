import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void softmax_forward_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int feature_idx = threadIdx.x;
    
    __shared__ scalar_t max_val;
    
    // Find the maximum value for each batch
    if (feature_idx == 0) {
        max_val = -INFINITY;
        for (int i = 0; i < dim; i++) {
            scalar_t val = input[batch_idx * dim + i];
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    __syncthreads();
    
    // Subtract max to prevent overflow
    scalar_t x = input[batch_idx * dim + feature_idx] - max_val;
    scalar_t exp_x = expf(x);
    
    // Compute the sum of exponentials
    __shared__ scalar_t sum;
    sum = 0.0;
    for (int stride = 1; stride <= dim; stride *= 2) {
        __syncthreads();
        if (feature_idx % (2 * stride) == 0) {
            sum += exp_x;
            if (feature_idx + stride < dim) {
                sum += expf(input[batch_idx * dim + feature_idx + stride] - max_val);
            }
        }
    }
    __syncthreads();
    
    // Write the result
    output[batch_idx * dim + feature_idx] = exp_x / sum;
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int dim = input.size(1);
    
    auto output = torch::empty_like(input);
    
    dim3 block_size(dim);
    dim3 grid_size(batch_size);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "softmax_forward", ([&] {
        softmax_forward_kernel<scalar_t><<<grid_size, block_size>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            dim
        );
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

softmax_header = """
torch::Tensor softmax_cuda(torch::Tensor input);
"""

softmax_op = load_inline(
    name="softmax_op",
    cpp_sources=softmax_header,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_cuda_op = softmax_op
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_cuda_op.softmax_cuda(x)