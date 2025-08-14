import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return argmax_cuda(x, self.dim)

argmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void argmax_kernel(const scalar_t* input, int64_t* output, int batch_size, int dim1, int dim2, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * (dim == 1 ? dim2 : dim1)) {
        return;
    }

    int max_val = -INT64_MAX;
    int max_idx = 0;

    if (dim == 0) {
        for (int i = 0; i < batch_size; i++) {
            int pos = i * dim1 * dim2 + idx % dim1 * dim2 + idx % dim2;
            int val = input[pos];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
    } else if (dim == 1) {
        for (int i = 0; i < dim1; i++) {
            int pos = idx / dim2 * batch_size * dim1 * dim2 + i * dim2 + idx % dim2;
            int val = input[pos];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
    } else { // dim == 2
        for (int i = 0; i < dim2; i++) {
            int pos = idx / dim2 * batch_size * dim1 * dim2 + (idx % dim1) * dim2 + i;
            int val = input[pos];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
    }

    output[idx] = max_idx;
}

std::vector<int64_t> compute_output_size(torch::Tensor input, int dim) {
    auto input_shape = input.sizes().vec();
    input_shape.erase(input_shape.begin() + dim);
    return input_shape;
}

torch::Tensor argmax_cuda(torch::Tensor input, int dim) {
    auto output_shape = compute_output_size(input, dim);
    auto output = torch::empty(output_shape, input.options().dtype(torch::kLong).device(torch::kCUDA));

    int batch_size = input.size(0);
    int dim1 = input.size(1);
    int dim2 = input.size(2);
    int total_elements = output.numel();

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    int device = input.get_device();
    cudaSetDevice(device);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmax_cuda", ([&] {
        argmax_kernel<scalar_t><<<num_blocks, block_size, 0, torch::cuda::current_stream().stream()>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            batch_size,
            dim1,
            dim2,
            dim
        );
    }));

    return output;
}
"""

argmax_cuda_header = """
std::vector<int64_t> compute_output_size(torch::Tensor input, int dim);
torch::Tensor argmax_cuda(torch::Tensor input, int dim);
"""

argmax_cuda = load_inline(
    name='argmax_cuda',
    cpp_sources=argmax_cuda_header,
    cuda_sources=argmax_cuda_source,
    functions=['argmax_cuda'],
    verbose=True,
    with_cuda=True
)

ModelNew.argmax_cuda = argmax_cuda