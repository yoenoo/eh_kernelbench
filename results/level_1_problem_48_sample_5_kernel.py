import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for mean reduction along a specific dimension
mean_reduction_cuda = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void mean_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size,
    int outer_dim,
    int inner_dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_dim * inner_dim) return;

    int outer = idx / inner_dim;
    int inner = idx % inner_dim;

    scalar_t sum = 0;
    for (int d = 0; d < dim_size; ++d) {
        int input_idx = outer * dim_size * inner_dim + d * inner_dim + inner;
        sum += input[input_idx];
    }
    output[idx] = sum / dim_size;
}

std::tuple<torch::Tensor> mean_reduction_cuda(torch::Tensor input, int dim) {
    auto input_size = input.sizes().vec();
    int input_ndim = input.dim();

    // Calculate the size of the dimension to reduce
    int dim_size = input.size(dim);

    // Compute the outer and inner dimensions
    int outer_dim = 1;
    for (int i = 0; i < dim; ++i) {
        outer_dim *= input.size(i);
    }
    int inner_dim = 1;
    for (int i = dim + 1; i < input_ndim; ++i) {
        inner_dim *= input.size(i);
    }

    auto output_size = input_size;
    output_size.erase(output_size.begin() + dim);
    auto output = torch::empty(output_size, input.options());

    int block_size = 256;
    int num_elements = outer_dim * inner_dim;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduction_cuda", ([&] {
        mean_reduction_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            outer_dim,
            inner_dim);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

# Compile the CUDA kernel
mean_reduction = load_inline(
    name="mean_reduction",
    cuda_sources=mean_reduction_cuda,
    functions=["mean_reduction_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mean_reduction.mean_reduction_cuda(x, self.dim)[0]

# Ensure the input generation matches the original
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]