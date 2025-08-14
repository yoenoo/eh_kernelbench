import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for exclusive cumulative sum
exclusive_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void exclusive_cumsum_kernel(const T* input, T* output, const int64_t* dims, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_size = dims[0];
    int dim_size = dims[1];

    if (idx < batch_size * dim_size) {
        int batch = idx / dim_size;
        int pos = idx % dim_size;
        T sum = T(0);
        for (int i = 0; i < pos; ++i) {
            sum += input[batch * dim_size + i];
        }
        output[idx] = sum;
    }
}

std::vector<int64_t> get_dimensions(torch::Tensor input, int dim) {
    auto sizes = input.sizes().vec();
    return {sizes[0], sizes[dim]};
}

torch::Tensor exclusive_cumsum_cuda(torch::Tensor input, int dim) {
    const auto dims = get_dimensions(input, dim);
    auto batch_size = dims[0];
    auto dim_size = dims[1];
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_elements = batch_size * dim_size;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "exclusive_cumsum_kernel", ([&] {
        exclusive_cumsum_kernel<scalar_t><<<num_blocks, block_size, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dims.data(),
            dim);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = """
torch::Tensor exclusive_cumsum_cuda(torch::Tensor input, int dim);
"""

exclusive_cumsum_ext = load_inline(
    name="exclusive_cumsum",
    cpp_sources=cpp_source,
    cuda_sources=exclusive_cumsum_source,
    functions=["exclusive_cumsum_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.exclusive_cumsum = exclusive_cumsum_ext

    def forward(self, x):
        return self.exclusive_cumsum.exclusive_cumsum_cuda(x, self.dim)