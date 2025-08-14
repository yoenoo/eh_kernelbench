import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

log_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

template <typename scalar_t>
__global__ void log_softmax_forward_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out, int batch_size, int dim_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Compute the maximum value in the specified dimension
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    for (int d = 0; d < dim_size; ++d) {
        scalar_t val = x[idx * dim_size + d];
        if (val > max_val) {
            max_val = val;
        }
    }

    // Compute the sum of exp(x_i - max)
    scalar_t sum = 0.0;
    for (int d = 0; d < dim_size; ++d) {
        sum += exp(x[idx * dim_size + d] - max_val);
    }

    scalar_t inv_sum = 1.0 / sum;
    for (int d = 0; d < dim_size; ++d) {
        out[idx * dim_size + d] = (x[idx * dim_size + d] - max_val) - log(sum);
    }
}

torch::Tensor log_softmax_forward_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int dim_size = x.size(1);
    auto out = torch::empty_like(x);

    const int threads_per_block = 256;
    const int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "log_softmax_forward_cuda", ([&] {
        log_softmax_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            dim_size,
            1  // Assuming dim is fixed as 1 per the model's default
        );
    }));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("log_softmax_forward_cuda", &log_softmax_forward_cuda, "LogSoftmax forward CUDA kernel");
}
"""

log_softmax_cpp_source = """
#include <torch/extension.h>
torch::Tensor log_softmax_forward_cuda(torch::Tensor x);
"""

log_softmax_extension = load_inline(
    name="log_softmax",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=["log_softmax_forward_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.log_softmax_forward = log_softmax_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_softmax_forward.log_softmax_forward_cuda(x)