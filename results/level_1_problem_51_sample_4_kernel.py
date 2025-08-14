import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmax_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

template <typename scalar_t>
__global__ void custom_argmax_kernel(const scalar_t* input, int64_t* output,
                                    int dim_size, int other_size,
                                    int dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // For each element in the non-reduced dimensions
    for (int i = index; i < other_size; i += stride) {
        // Compute the offset along the reduced dimension
        int max_idx = 0;
        scalar_t max_val = -INFINITY;
        for (int d = 0; d < dim_size; ++d) {
            // Compute the position in the input tensor
            int pos = i * dim_size + d;
            scalar_t val = input[pos];
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }
        output[i] = max_idx;
    }
}

std::vector<int64_t> compute_output_shape(torch::Tensor input, int dim) {
    auto in_shape = input.sizes().vec();
    in_shape.erase(in_shape.begin() + dim);
    return in_shape;
}

torch::Tensor custom_argmax(torch::Tensor input, int dim) {
    auto input_shape = input.sizes();
    int input_dim = input.dim();
    if (dim < 0) dim += input_dim;
    auto output_shape = compute_output_shape(input, dim);
    auto output = torch::empty(output_shape, input.options().dtype(torch::kLong));

    int other_size = output.numel();
    int dim_size = input.size(dim);

    int threads = 256;
    int blocks = (other_size + threads - 1) / threads;

    const int cta = blocks * threads;
    dim3 grid(blocks);
    dim3 block(threads);

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "custom_argmax", ([&] {
        custom_argmax_kernel<scalar_t><<<grid, block>>>(
            input.data<scalar_t>(),
            output.data_ptr<int64_t>(),
            dim_size, other_size, dim);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

argmax_cpp_source = """
#include <torch/extension.h>

torch::Tensor custom_argmax(torch::Tensor input, int dim);
"""

argmax_cuda = load_inline(name="argmax_cuda",
                         cpp_sources=argmax_cpp_source,
                         cuda_sources=argmax_source,
                         functions=["custom_argmax"],
                         verbose=True)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax_op = argmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmax_op.custom_argmax(x, self.dim)