import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmax
argmax_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void argmax_kernel(const scalar_t* input, int* output,
                             int dim_size, int outer_size, int inner_size,
                             int64_t dim) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= outer_size * inner_size) {
        return;
    }

    const int outer = index / inner_size;
    const int inner = index % inner_size;
    const int offset = outer * dim_size + inner;

    int max_idx = 0;
    scalar_t max_val = input[offset];
    for (int d = 0; d < dim_size; ++d) {
        scalar_t val = input[offset + d * inner_size];
        if (val > max_val) {
            max_val = val;
            max_idx = d;
        }
    }
    output[outer * inner_size + inner] = max_idx;
}

std::vector<int> compute_gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return {a};
}

torch::Tensor argmax_cuda(torch::Tensor input, int64_t dim) {
    const int64_t* sizes = input.sizes().data();
    int ndim = input.dim();
    int64_t dim_size = sizes[dim];

    // Calculate the dimension sizes for the output tensor
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            out_sizes.push_back(sizes[i]);
        }
    }

    torch::Tensor output = torch::empty(out_sizes, input.options().dtype(torch::kInt32)).cuda();

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= sizes[i];
    }

    const int block_size = 256;
    const int num_blocks = (outer_size * inner_size + block_size - 1) / block_size;

    // Determine the scalar type and launch kernel
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "argmax_cuda", [&] {
        argmax_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data<scalar_t>(),
            output.data_ptr<int>(),
            dim_size,
            outer_size,
            inner_size,
            dim);
    });

    return output;
}
"""

argmax_cpp_source = """
torch::Tensor argmax_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code for argmax
argmax_op = load_inline(
    name="argmax_op",
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_kernel_source,
    functions=["argmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax = argmax_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call custom CUDA argmax implementation
        return self.argmax.argmax_cuda(x, self.dim).to(x.device)