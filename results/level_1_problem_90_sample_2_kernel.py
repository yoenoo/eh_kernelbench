import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cumulative product
cumprod_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void cumulative_product_kernel(const scalar_t* input, scalar_t* output, int dim_size, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int element_idx = threadIdx.x;

    // Compute the index along the dimension
    int index = batch_idx * dim_size + element_idx;

    // Each thread handles one element in the dimension
    if (element_idx == 0) {
        output[index] = input[index];
    } else {
        // Ensure previous element is computed before this one
        __threadfence_block();
        output[index] = output[index - 1] * input[index];
    }
}

torch::Tensor cumulative_product_cuda(torch::Tensor input, int dim) {
    const int batch_size = input.size(0);
    const int dim_size = input.size(1);
    auto output = torch::empty_like(input);

    dim3 blocks(batch_size);
    dim3 threads(dim_size);

    // Launch one block per batch element. Each block processes a dimension.
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "cumulative_product_cuda", ([&] {
        using scalar_t = scalar_type;
        cumulative_product_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), dim_size, batch_size, dim);
    }));

    return output;
}
"""

cumprod_header = """
torch::Tensor cumulative_product_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
cumprod_module = load_inline(
    name="cumprod",
    cpp_sources=cumprod_header,
    cuda_sources=cumprod_source,
    functions=["cumulative_product_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumulative_product_cuda = cumprod_module.cumulative_product_cuda

    def forward(self, x):
        return self.cumulative_product_cuda(x, self.dim)

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]

# Define input dimensions and parameters
batch_size = 32768
input_shape = (32768,)
dim = 1