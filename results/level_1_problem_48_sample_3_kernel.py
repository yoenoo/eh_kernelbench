import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void mean_reduction_kernel(const scalar_t* __restrict__ input,
                                     scalar_t* __restrict__ output,
                                     const int total_elements,
                                     const int output_size,
                                     const int reduce_dim,
                                     const int dim_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_size) return;

    scalar_t sum = 0.0;
    int batch_stride = 1;
    for (int d = input.dim() - 1; d > reduce_dim; --d) {
        batch_stride *= input.size(d);
    }
    int reduce_stride = 1;
    for (int d = reduce_dim - 1; d >= 0; --d) {
        reduce_stride *= input.size(d);
    }
    
    // Iterate through each element in the reduced dimension
    for (int r = 0; r < dim_size; ++r) {
        int pos = index * reduce_stride + r * batch_stride;
        sum += input[pos];
    }
    output[index] = sum / dim_size;
}

torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim) {
    const auto input_size = input.sizes().vec();
    int input_dim = input.dim();
    int dim_size = input.size(dim);

    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    const int total_elements = input.numel();
    const int output_size = output.numel();
    const int threads = 256;
    const int blocks = (output_size + threads - 1) / threads;

    // Determine dimensions using passed dim
    // Assuming input is contiguous
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduction_cuda", ([&] {
        mean_reduction_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total_elements,
            output_size,
            dim,
            dim_size
        );
    }));

    return output;
}

"""

mean_reduction_cpp_source = "torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim);"

mean_reduction = load_inline(
    name="mean_reduction",
    cpp_sources=mean_reduction_cpp_source,
    cuda_sources=mean_reduction_source,
    functions=["mean_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mean_reduction = mean_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_reduction.mean_reduction_cuda(x, self.dim)