import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Define the custom CUDA kernel for reverse cumulative sum
        reverse_cumsum_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void reverse_cumsum_kernel(const scalar_t* input, scalar_t* output, const int dim, const int size, const int inner_dim, const int outer_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int outer = idx / inner_dim;
    int inner = idx % inner_dim;

    // Compute the reverse index along the dimension
    int original_pos = outer * inner_dim + (inner_dim - 1 - inner);
    int output_pos = outer * inner_dim + inner;

    // Accumulate in reverse order
    scalar_t sum = 0;
    for (int i = 0; i <= inner; i++) {
        int input_pos = outer * inner_dim + (inner_dim - 1 - i);
        sum += input[input_pos];
    }
    output[output_pos] = sum;
}

torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    auto dims = input.sizes().vec();
    int total_elements = input.numel();
    
    // Calculate the inner and outer dimensions
    int inner_dim = 1;
    for (int i = dim + 1; i < dims.size(); i++)
        inner_dim *= dims[i];
    int outer_dim = 1;
    for (int i = 0; i < dim; i++)
        outer_dim *= dims[i];
    
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    // Launch the kernel with appropriate dimensions
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "reverse_cumsum_cuda", ([&] {
        reverse_cumsum_kernel<scalar_t><<<grid_size, block_size>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            dim,
            total_elements,
            inner_dim,
            outer_dim
        );
    }));

    return output;
}
"""

        reverse_cumsum_cuda_header = """
torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int64_t dim);
"""

        # Compile the CUDA kernel
        self.reverse_cumsum = load_inline(
            name='reverse_cumsum',
            cpp_sources=reverse_cumsum_cuda_header,
            cuda_sources=reverse_cumsum_cuda_source,
            functions=['reverse_cumsum_cuda'],
            verbose=False
        )

    def forward(self, x):
        return self.reverse_cumsum.reverse_cumsum_cuda(x, self.dim)