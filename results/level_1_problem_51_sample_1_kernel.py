import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <limits>

template <typename scalar_t>
__global__ void argmax_kernel(const scalar_t* input, int* output,
                             int dim_size, int outer_size, int inner_size,
                             int dim) {
    int batch_idx = blockIdx.x / inner_size;
    int inner_idx = blockIdx.x % inner_size;

    int index = batch_idx * inner_size * dim_size + inner_idx * dim_size;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_idx = 0;

    for (int d = 0; d < dim_size; ++d) {
        scalar_t val = input[index + d];
        if (val > max_val) {
            max_val = val;
            max_idx = d;
        }
    }

    int out_index = batch_idx * inner_size + inner_idx;
    output[out_index] = max_idx;
}

std::vector<int> compute_grid_dims(int batch_size, int dim1, int dim2, int dim) {
    int outer_size = 1;
    int inner_size = 1;
    if (dim == 0) {
        outer_size = 1;
        inner_size = dim1 * dim2;
    } else if (dim == 1) {
        outer_size = batch_size;
        inner_size = dim2;
    } else if (dim == 2) {
        outer_size = batch_size * dim1;
        inner_size = 1;
    }

    int grid_size = outer_size * inner_size;
    return {grid_size, 1, 1};
}

torch::Tensor argmax_cuda(torch::Tensor input, int dim) {
    const auto batch_size = input.size(0);
    const auto dim1 = input.size(1);
    const auto dim2 = input.size(2);

    auto output = torch::empty({batch_size, dim == 0 ? 1 : (dim == 1 ? dim2 : dim1)}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    auto grid_dims = compute_grid_dims(batch_size, dim1, dim2, dim);
    dim3 blocks(grid_dims[0]);
    dim3 threads(1);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmax_cuda", ([&] {
        argmax_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data_ptr<int>(),
            input.size(dim),
            grid_dims[1],
            grid_dims[2],
            dim);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

argmax_cpp_source = """
torch::Tensor argmax_cuda(torch::Tensor input, int dim);
"""

argmax_extension = load_inline(
    name='argmax_cuda',
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_cuda_source,
    functions=['argmax_cuda'],
    verbose=True
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return argmax_extension.argmax_cuda(x, self.dim).to(x.device)