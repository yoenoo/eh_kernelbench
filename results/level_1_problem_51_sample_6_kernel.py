import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

template <typename scalar_t>
__global__ void argmax_kernel(const scalar_t* input, int64_t* output,
                             int batch_size, int dim1, int dim2, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * ((dim == 0) ? 1 : (dim == 1 ? dim2 : dim1))) {
        return;
    }

    int batch, d1, d2;
    if (dim == 0) {
        batch = 0;
        d1 = idx / dim2;
        d2 = idx % dim2;
    } else if (dim == 1) {
        batch = idx / dim2;
        d1 = 0;
        d2 = idx % dim2;
    } else {  // dim == 2
        batch = idx / (dim1);
        d1 = idx % (dim1);
        d2 = 0;
    }

    scalar_t max_val = -INFINITY;
    int max_idx = 0;
    for (int i = 0; i < (dim == 0 ? batch_size : (dim == 1 ? dim1 : dim2)); ++i) {
        int input_idx;
        if (dim == 0) {
            input_idx = i * dim1 * dim2 + d1 * dim2 + d2;
        } else if (dim == 1) {
            input_idx = batch * dim1 * dim2 + i * dim2 + d2;
        } else {
            input_idx = batch * dim1 * dim2 + d1 * dim2 + i;
        }
        scalar_t val = input[input_idx];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    output[idx] = max_idx;
}

torch::Tensor argmax_cuda(torch::Tensor input, int dim) {
    auto batch_size = input.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    auto output_size = input.sizes().vec();
    output_size.erase(output_size.begin() + dim);
    auto output = torch::empty(output_size, input.options().dtype(torch::kLong));

    int threads = 256;
    dim3 blocks((output.numel() + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmax_cuda", ([&] {
        argmax_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data<int64_t>(),
            batch_size, dim1, dim2, dim);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

argmax_cpp_source = "torch::Tensor argmax_cuda(torch::Tensor input, int dim);"

argmax_op = load_inline(
    name="argmax_op",
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_source,
    functions=["argmax_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.argmax_op = argmax_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmax_op.argmax_cuda(x, self.dim)