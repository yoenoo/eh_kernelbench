import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void max_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2,
    const int reduce_dim,
    const int outer_dim,
    const int inner_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_dim * inner_dim) {
        return;
    }

    int outer = idx / inner_dim;
    int inner = idx % inner_dim;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    if (reduce_dim == 1) {
        for (int i = 0; i < dim1; ++i) {
            int input_idx = outer * dim1 * dim2 + i * dim2 + inner;
            max_val = fmax(max_val, input[input_idx]);
        }
    } else if (reduce_dim == 2) {
        for (int i = 0; i < dim2; ++i) {
            int input_idx = outer * dim1 * dim2 + inner * dim2 + i;
            max_val = fmax(max_val, input[input_idx]);
        }
    }

    output[outer * inner_dim + inner] = max_val;
}

torch::Tensor max_reduction_cuda(torch::Tensor input, int dim) {
    const int batch_size = input.size(0);
    const int reduce_dim = dim + 1; // because in CUDA, dimensions are 0-based
    int outer_dim, inner_dim;
    int dim1 = input.size(1);
    int dim2 = input.size(2);

    if (reduce_dim == 1) { // reducing over dim1
        outer_dim = batch_size;
        inner_dim = dim2;
    } else { // reducing over dim2
        outer_dim = batch_size * dim1;
        inner_dim = 1; // since dim2 is the reduced dimension
    }

    const int total_threads = outer_dim * inner_dim;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;

    auto output = torch::empty({batch_size, (reduce_dim == 1 ? 1 : dim1), (reduce_dim == 1 ? dim2 : 1)}, 
                             torch::device(input.device()).dtype(input.scalar_type()));

    max_reduction_kernel<<<num_blocks, block_size, 0, 
        torch::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<scalar_t>(), 
        output.data_ptr<scalar_t>(),
        batch_size,
        dim1,
        dim2,
        reduce_dim,
        outer_dim,
        inner_dim
    );

    return output;
}
"""

max_reduction_cpp_source = """
torch::Tensor max_reduction_cuda(torch::Tensor input, int dim);
"""

max_reduction = load_inline(
    name="max_reduction",
    cpp_sources=max_reduction_cpp_source,
    cuda_sources=max_reduction_source,
    functions=["max_reduction_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.max_reduction = max_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_reduction.max_reduction_cuda(x, self.dim).squeeze()