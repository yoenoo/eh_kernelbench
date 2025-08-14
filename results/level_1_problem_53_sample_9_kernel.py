import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void min_reduction_kernel(
    const float* input,
    float* output,
    int batch_size,
    int dim1,
    int dim2,
    int reduce_dim
) {
    int batch_idx = blockIdx.x;
    int output_idx = threadIdx.x;

    if (batch_idx >= batch_size || output_idx >= (reduce_dim == 0 ? dim1 : dim2)) {
        return;
    }

    int input_idx;
    if (reduce_dim == 0) {
        input_idx = batch_idx * dim1 * dim2 + output_idx * dim2;
    } else if (reduce_dim == 1) {
        input_idx = batch_idx * dim1 * dim2 + output_idx;
    } else {
        // Handle other dimensions if needed
        return;
    }

    float min_val = input[input_idx];
    int step = (reduce_dim == 0) ? 1 : dim1;

    for (int i = 0; i < (reduce_dim == 0 ? dim2 : dim1); ++i) {
        min_val = min(min_val, input[input_idx + i * step]);
    }

    output[batch_idx * (reduce_dim == 0 ? dim2 : dim1) + output_idx] = min_val;
}

torch::Tensor min_reduction_cuda(
    torch::Tensor input,
    int dim
) {
    int batch_size = input.size(0);
    int reduce_dim = dim;
    int dim1 = input.size(1);
    int dim2 = input.size(2);

    int output_size_0 = batch_size;
    int output_size_1 = (reduce_dim == 0) ? dim2 : dim1;

    auto output = torch::empty({output_size_0, output_size_1}, torch::device("cuda"));

    dim3 blocks(batch_size);
    dim3 threads((reduce_dim == 0) ? dim1 : dim2);

    min_reduction_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim1,
        dim2,
        reduce_dim
    );

    return output;
}
"""

min_reduction_cpp_source = """
torch::Tensor min_reduction_cuda(torch::Tensor input, int dim);
"""

min_reduction = load_inline(
    name="min_reduction",
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_source,
    functions=["min_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.min_reduction = min_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_reduction.min_reduction_cuda(x, self.dim)