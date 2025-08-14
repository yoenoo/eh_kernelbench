import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template<typename scalar_t>
__global__ void mean_reduction_kernel(const scalar_t* __restrict__ input,
                                     scalar_t* __restrict__ output,
                                     int dim_size,
                                     int outer_size,
                                     int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size * inner_size) {
        return;
    }

    int outer = idx / inner_size;
    int inner = idx % inner_size;

    scalar_t sum = 0;
    for (int d = 0; d < dim_size; ++d) {
        int input_idx = outer * dim_size * inner_size + d * inner_size + inner;
        sum += input[input_idx];
    }
    output[idx] = sum / dim_size;
}

torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim) {
    auto input_size = input.sizes().vec();
    int dim_size = input.size(dim);
    auto output_size = input_sizes;
    output_size[dim] = 1;
    auto output = torch::empty(output_size, input.options());

    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= input.size(i);
    }
    int inner_size = 1;
    for (int i = dim + 1; i < input.dim(); ++i) {
        inner_size *= input.size(i);
    }

    const int block_size = 256;
    const int num_elements = outer_size * inner_size;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    mean_reduction_kernel<float><<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim_size,
        outer_size,
        inner_size
    );

    return output.squeeze(dim);
}
"""

mean_reduction_cpp_source = "torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim);"

mean_reduction = load_inline(
    name="mean_reduction",
    cpp_sources=mean_reduction_cpp_source,
    cuda_sources=mean_reduction_source,
    functions=["mean_reduction_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mean_reduction.mean_reduction_cuda(x, self.dim)