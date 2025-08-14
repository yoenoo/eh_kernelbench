import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmax
argmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

template <typename scalar_t>
__global__ void argmax_kernel(const scalar_t* input, int64_t* output, int64_t dim_size, int64_t outer_size, int64_t inner_size, int64_t dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size * inner_size) return;

    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;

    int64_t offset = outer * dim_size + inner * dim_size;
    scalar_t max_val = -INFINITY;
    int64_t max_idx = 0;
    for (int64_t d = 0; d < dim_size; ++d) {
        scalar_t val = input[offset + d];
        if (val > max_val) {
            max_val = val;
            max_idx = d;
        }
    }
    output[outer * inner_size + inner] = max_idx;
}

std::tuple<torch::Tensor, torch::Tensor> argmax_cuda(torch::Tensor input, int64_t dim) {
    const int64_t ndim = input.dim();
    int64_t target_dim = dim;

    // Compute tensor dimensions
    int64_t outer_size = 1;
    for (int64_t i = 0; i < target_dim; ++i) {
        outer_size *= input.size(i);
    }
    int64_t dim_size = input.size(target_dim);
    int64_t inner_size = 1;
    for (int64_t i = target_dim + 1; i < ndim; ++i) {
        inner_size *= input.size(i);
    }
    
    // Output tensor
    auto output_shape = input.sizes().vec();
    output_shape.erase(output_shape.begin() + target_dim);
    auto output = torch::empty(output_shape, input.options().dtype(torch::kLong));

    // Launch kernel
    const int block_size = 256;
    const int num_elements = outer_size * inner_size;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    // The kernel will now handle the computation
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmax_cuda", ([&] {
        argmax_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data<scalar_t>(),
            output.data_ptr<int64_t>(),
            dim_size,
            outer_size,
            inner_size,
            target_dim);
        });
    cudaDeviceSynchronize();

    return std::make_tuple(output, output); // Return output twice to match torch.argmax output signature
}

"""

argmax_cpp_source = (
    "at::Tensor elementwise_add_cuda(const at::Tensor a, const at::Tensor b);"
)

# Compile the inline CUDA code for argmax
argmax_module = load_inline(
    name="argmax_op",
    cpp Sources="",
    cuda_sources=argmax_source,
    functions=["argmax_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax_op = argmax_module.argmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.argmax_op(x, self.dim)
        return out