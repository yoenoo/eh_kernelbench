import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void softmax_forward_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, const int batch_size, const int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const int start = idx * dim;
    const int end = start + dim;

    // Compute max value for stability
    scalar_t max_val = -INFINITY;
    for (int i = start; i < end; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    scalar_t sum = 0;
    for (int i = start; i < end; ++i) {
        scalar_t exp_x = exp(input[i] - max_val);
        output[i] = exp_x;
        sum += exp_x;
    }

    for (int i = start; i < end; ++i) {
        output[i] /= sum;
    }
}

torch::Tensor softmax_forward_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int batch_size = input.size(0);
    const int dim = input.size(1);
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softmax_forward_cuda", ([&] {
        softmax_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim);
    }));

    return output;
}
"""

softmax_cpp_source = "torch::Tensor softmax_forward_cuda(torch::Tensor input);"

softmax_cuda = load_inline(
    name="softmax_cuda",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_cuda_source,
    functions=["softmax_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax = softmax_cuda

    def forward(self, x):
        return self.softmax.softmax_forward_cuda(x)