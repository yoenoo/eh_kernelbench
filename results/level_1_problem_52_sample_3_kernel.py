import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmin_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void argmin_kernel(const scalar_t* data, int* output,
                             int batch_size, int dim1, int dim2, int dim) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int outer_dim = dim == 0 ? 1 : dim1;
    int inner_dim = dim == 0 ? dim1 * dim2 : dim2;
    int outer_size = outer_dim;
    int inner_size = inner_dim;

    for (int i = 0; i < outer_size; ++i) {
        scalar_t min_val = std::numeric_limits<scalar_t>::max();
        int min_idx = 0;
        int index_offset = batch_idx * dim1 * dim2 + i * inner_size;
        for (int j = 0; j < inner_size; ++j) {
            scalar_t val = data[index_offset + j];
            if (val < min_val) {
                min_val = val;
                min_idx = j;
            }
        }
        output[batch_idx * outer_size + i] = min_idx;
    }
}

torch::Tensor argmin_cuda(torch::Tensor data, int dim) {
    int batch_size = data.size(0);
    int dim1 = data.size(1);
    int dim2 = data.size(2);

    auto output = torch::empty({batch_size, dim == 0 ? 1 : dim1}, torch::dtype(torch::kInt32).device(data.device()));

    int block_size = 256;
    int grid_size = (batch_size + block_size - 1) / block_size;

    dim3 blocks(grid_size);
    dim3 threads(block_size);

    AT_DISPATCH_FLOATING_TYPES(data.scalar_type(), "argmin_cuda", ([&] {
        argmin_kernel<scalar_t><<<blocks, threads>>>(
            data.data<scalar_t>(), output.data<int>(),
            batch_size, dim1, dim2, dim);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

argmin_cuda_header = """
torch::Tensor argmin_cuda(torch::Tensor data, int dim);
"""

argmin_cuda = load_inline(
    name="argmin_cuda",
    cpp_sources=argmin_cuda_header,
    cuda_sources=argmin_cuda_source,
    functions=["argmin_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmin_cuda = argmin_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmin_cuda.argmin_cuda(x, self.dim).to(x.device)