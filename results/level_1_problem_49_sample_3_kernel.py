import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_reduction_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void max_reduction_kernel(const scalar_t* __restrict__ input,
                                    scalar_t* __restrict__ output,
                                    int batch_size,
                                    int dim1,
                                    int dim2,
                                    int reduce_dim) {
    int batch_idx = blockIdx.x;
    int output_idx = batch_idx * dim2;

    // Initialize max value to the smallest possible
    scalar_t max_val = -FLT_MAX;

    // Iterate over the elements along the reduction dimension (dim1)
    for (int i = 0; i < dim1; ++i) {
        int input_idx = batch_idx * dim1 * dim2 + i * dim2;
        scalar_t val = input[input_idx];
        if (val > max_val) {
            max_val = val;
        }
    }

    output[output_idx] = max_val;
}

torch::Tensor max_reduction_cuda(torch::Tensor input, int dim) {
    const int batch_size = input.size(0);
    const int dim1 = input.size(1);
    const int dim2 = input.size(2);

    auto output = torch::empty({batch_size, dim2}, input.options());

    const int blocks = batch_size;
    const int threads = 1; // Since each block handles one batch element

    if (dim != 1) {
        // For this simple example, kernel is written to handle reduction over dim=1 only
        // Modify the kernel if reduction over other dimensions is needed
        AT_ERROR("Kernel currently supports reduction over dim=1 only");
    }

    max_reduction_kernel<float><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size,
        dim1,
        dim2,
        dim);

    return output;
}
"""

max_reduction_cuda_header = """
torch::Tensor max_reduction_cuda(torch::Tensor input, int dim);
"""

max_reduction_ext = load_inline(
    name='max_reduction_cuda',
    cpp_sources=max_reduction_cuda_header,
    cuda_sources=max_reduction_cuda_source,
    functions=['max_reduction_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.max_reduction_cuda = max_reduction_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The current kernel is designed for reduction over dim=1. Modify as needed.
        if self.dim != 1:
            # Fallback to PyTorch's implementation if dim is not 1
            return torch.max(x, dim=self.dim)[0]
        else:
            return self.max_reduction_cuda.max_reduction_cuda(x, self.dim)

# The get_inputs and get_init_inputs functions remain unchanged
def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]  # Example, change to desired dimension