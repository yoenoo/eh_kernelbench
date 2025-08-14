import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

template <typename scalar_t>
__global__ void softmax_forward_kernel(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      const int batch_size,
                                      const int dim) {
    const int batch_idx = blockIdx.x;
    const int feature_idx = threadIdx.x;

    __shared__ scalar_t max_val;

    if (feature_idx == 0) {
        max_val = -INFINITY;
        for (int i = 0; i < dim; ++i) {
            scalar_t val = input[batch_idx * dim + i];
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    __syncthreads();

    scalar_t numerator = exp(input[batch_idx * dim + feature_idx] - max_val);
    scalar_t denominator = 0.0;

    // Using parallel reduction to compute the sum in denominator
    __shared__ scalar_t shared_buf[1024];
    shared_buf[threadIdx.x] = numerator;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (threadIdx.x < stride) {
            shared_buf[threadIdx.x] += shared_buf[threadIdx.x + stride];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        denominator = shared_buf[0];
        for (int i = stride; i < dim; ++i) { // Accumulate remaining elements if any
            denominator += shared_buf[i];
        }
    }
    __syncthreads();

    output[batch_idx * dim + feature_idx] = numerator / denominator;
}

torch::Tensor softmax_forward(torch::Tensor input) {
    const auto batch_size = input.size(0);
    const auto dim = input.size(1);

    auto output = torch::empty({batch_size, dim}, input.options());

    const int block_size = 512; // Tuned for best performance
    dim3 grid(batch_size);
    dim3 block(std::min(dim, block_size)); // Ensure block size <= 1024

    // Launch kernel based on input dtype
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softmax_forward", ([&] {
        softmax_forward_kernel<scalar_t><<<grid, block>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            dim);
    }));

    return output;
}
"""

softmax_cpp_source = "torch::Tensor softmax_forward(torch::Tensor input);"

softmax_ext = load_inline(
    name="softmax_cuda",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_forward"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-lineinfo"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax = softmax_ext

    def forward(self, x):
        return self.softmax.softmax_forward(x.cuda())