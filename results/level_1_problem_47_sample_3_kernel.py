import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for sum reduction
sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sum_reduction_kernel(const scalar_t* input, scalar_t* output, int dim_size, int outer_size, int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size * inner_size) return;

    int outer = idx / inner_size;
    int inner = idx % inner_size;
    scalar_t sum = 0;
    for (int d = 0; d < dim_size; ++d) {
        sum += input[outer * dim_size * inner_size + d * inner_size + inner];
    }
    output[idx] = sum;
}

std::vector<torch::Tensor> sum_reduction_cuda(torch::Tensor input, int dim, bool keepdim) {
    int ndim = input.dim();
    int dim_size = input.size(dim);
    auto input_shape = input.sizes().vec();
    auto output_shape = input.sizes().vec();
    if (keepdim) {
        output_shape[dim] = 1;
    } else {
        output_shape.erase(output_shape.begin() + dim);
    }
    auto output = torch::empty(output_shape, input.options());
    
    // Calculate outer and inner dimensions
    int dim_idx = dim;
    int outer_dim = 1;
    for (int i = 0; i < dim_idx; i++) {
        outer_dim *= input.size(i);
    }
    int inner_dim = 1;
    for (int i = dim_idx + 1; i < ndim; i++) {
        inner_dim *= input.size(i);
    }
    
    const int block_size = 256;
    int num_elements = outer_dim * inner_dim;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    dim3 blocks(num_blocks);
    dim3 threads(block_size);

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "sum_reduction_cuda", ([&] {
        sum_reduction_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            dim_size,
            outer_dim,
            inner_dim
        );
    }));

    return {output};
}
"""

sum_reduction_cpp_source = """
std::vector<torch::Tensor> sum_reduction_cuda(torch::Tensor input, int dim, bool keepdim);
"""

# Compile the inline CUDA code for sum reduction
sum_reduction = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_cpp_source,
    cuda_sources=sum_reduction_source,
    functions=["sum_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.keepdim = True  # Since the original model uses keepdim=True
        self.sum_reduction = sum_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.sum_reduction.sum_reduction_cuda(x, self.dim, self.keepdim)
        return outputs[0]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [reduce_dim]