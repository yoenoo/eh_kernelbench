import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void reverse_cumsum_kernel(const float* input, float* output, int dim_size, int batch_size, int dim, int total_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_size) return;

    int dim_offset = index / dim_size;
    int pos = index % dim_size;

    float sum = 0;
    for (int i = 0; i <= pos; ++i) {
        int flip_pos = dim_size - 1 - i;
        int input_idx = dim_offset * dim_size + flip_pos;
        sum += input[input_idx];
    }
    output[index] = sum;
}

torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim) {
    auto dims = input.sizes().vec();
    int total_size = input.numel();
    int dim_size = dims[dim];
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;

    reverse_cumsum_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim_size,
        dims[0],  // batch_size
        dim,
        total_size
    );

    return output;
}
"""

reverse_cumsum_cpp_source = (
    "torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim);"
)

reverse_cumsum = load_inline(
    name="reverse_cumsum",
    cpp_sources=reverse_cumsum_cpp_source,
    cuda_sources=reverse_cumsum_source,
    functions=["reverse_cumsum_cuda"],
    verbose=True,
    extra_cflags=["-DILLBACKWARD_GPU"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reverse_cumsum_cuda = reverse_cumsum

    def forward(self, x):
        return self.reverse_cumsum_cuda.reverse_cumsum_cuda(x, self.dim)