import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

reverse_cumsum_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void reverse_cumsum_kernel(scalar_t* out, const scalar_t* in, int dim_size, int total_size, int dimStride, int batch_size, int inner_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_size) {
        // Compute the position along the dimension
        int pos = (index / inner_size) % dim_size;
        // Compute the reversed position
        int rev_pos = dim_size - 1 - pos;
        // Compute offset to the reversed position
        int rev_index = index - pos * inner_size + rev_pos * inner_size;
        // Accumulate from the end towards the current position
        scalar_t sum = 0;
        for (int i = rev_pos; i >= 0; --i) {
            int current_idx = index - (rev_pos - i) * inner_size;
            sum += in[current_idx];
        }
        out[rev_index] = sum;
    }
}

std::vector<int64_t> get_strides(const torch::Tensor& x) {
    auto sizes = x.sizes();
    int64_t stride = 1;
    std::vector<int64_t> strides(sizes.size());
    for (size_t i = sizes.size(); i --> 0; ) {
        strides[i] = stride;
        stride *= sizes[i];
    }
    return strides;
}

torch::Tensor reverse_cumsum_cuda(torch::Tensor x, int64_t dim) {
    auto x_size = x.sizes();
    int64_t total_elements = x.numel();
    int64_t dim_size = x.size(dim);
    auto strides = get_strides(x);
    int64_t dimStride = strides[dim]; // Actually the inner size after dim?

    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    auto out = torch::empty_like(x);

    const int batch_size = x_size[0];
    const int inner_size = 1;
    for (size_t i = dim + 1; i < x.dim(); ++i) {
        inner_size *= x_size[i];
    }

    reverse_cumsum_kernel<float><<<grid_size, block_size>>>(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        dim_size,
        total_elements,
        dimStride,
        batch_size,
        inner_size
    );

    return out;
}
"""

reverse_cumsum_cuda_header = """
std::vector<int64_t> get_strides(const torch::Tensor& x);
torch::Tensor reverse_cumsum_cuda(torch::Tensor x, int64_t dim);
"""

reverse_cumsum = load_inline(
    name="reverse_cumsum",
    cpp_sources=reverse_cumsum_cuda_header,
    cuda_sources=reverse_cumsum_cuda_source,
    functions=["reverse_cumsum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reverse_cumsum = reverse_cumsum

    def forward(self, x):
        return self.reverse_cumsum.reverse_cumsum_cuda(x, self.dim)