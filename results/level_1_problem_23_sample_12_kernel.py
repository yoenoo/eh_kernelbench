import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

template <typename scalar_t>
__device__ scalar_t logsumexp(scalar_t* data, int dim_size) {
    scalar_t max_val = data[0];
    for (int i = 1; i < dim_size; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    scalar_t sum = 0;
    for (int i = 0; i < dim_size; ++i) {
        sum += exp(data[i] - max_val);
    }
    return max_val + log(sum);
}

template <typename scalar_t>
__global__ void fast_softmax_forward(
    scalar_t* out,
    const scalar_t* in,
    int batch_size,
    int dim_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    scalar_t* data = in + idx * dim_size;
    scalar_t max_val = data[0];
    for (int i = 1; i < dim_size; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    scalar_t sum = 0;
    for (int i = 0; i < dim_size; ++i) {
        sum += exp(data[i] - max_val);
    }
    scalar_t lse = log(sum) + max_val;

    for (int i = 0; i < dim_size; ++i) {
        out[idx * dim_size + i] = exp(data[i] - lse);
    }
}

at::Tensor fast_softmax_cuda(at::Tensor input) {
    at::cuda::OptionalCUDAGuard device_guard(input.device());
    int batch_size = input.size(0);
    int dim_size = input.size(1);

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({batch_size, dim_size}, options);

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fast_softmax_cuda", ([&]{
        using scalar_t = scalar_t;
        fast_softmax_forward<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            batch_size,
            dim_size);
    }));

    return output;
}
"""

softmax_forward = load_inline(
    name="fast_softmax",
    cpp_sources="",
    cuda_sources=softmax_source,
    functions=["fast_softmax_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_func = softmax_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_func.fast_softmax_cuda(x)