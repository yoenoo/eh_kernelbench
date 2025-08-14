import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmin_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

template <typename scalar_t>
__global__ void argmin_kernel(const scalar_t* data, int64_t* indices,
    int batch_size, int dim1, int dim2, int dim) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;

    int stride = dim == 0 ? 1 : (dim == 1 ? dim2 : batch_size * dim1 * dim2);
    int max_size = (dim == 0) ? 1 : (dim == 1 ? dim1 : dim2);
    
    for (int i = 0; i < max_size; ++i) {
        int index = batch_idx * dim1 * dim2 + 
            (dim == 1 ? i * dim2 : 0) + 
            (dim == 2 ? i : 0);
        scalar_t min_val = data[index];
        int min_idx = 0;
        
        for (int j = 0; j < max_size; j++) {
            int current_idx = batch_idx * dim1 * dim2 + 
                (dim == 1 ? i * dim2 : j * dim1 * dim2) + 
                (dim == 2 ? j : 0);
            if (data[current_idx] < min_val) {
                min_val = data[current_idx];
                min_idx = j;
            }
        }
        
        indices[batch_idx * max_size + i] = min_idx;
    }
}

torch::Tensor argmin_cuda(torch::Tensor data, int dim) {
    int64_t batch_size = data.size(0);
    int dim1 = data.size(1);
    int dim2 = data.size(2);
    
    auto options = torch::TensorOptions().like(data).dtype(torch::kLong);
    torch::Tensor indices = torch::empty({batch_size, dim == 1 ? dim1 : (dim == 0 ? 1 : dim2)}, options);
    
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    AT_DISPATCH_ALL_TYPES(data.scalar_type(), "argmin_cuda", ([&] {
        argmin_kernel<scalar_t><<<num_blocks, block_size>>>(
            data.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            batch_size, dim1, dim2, dim);
    }));

    return indices;
}
"""

argmin_cpp_source = "torch::Tensor argmin_cuda(torch::Tensor data, int dim);"

argmin = load_inline(
    name="argmin_op",
    cpp_sources=argmin_cpp_source,
    cuda_sources=argmin_source,
    functions=["argmin_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmin_op = argmin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmin_op.argmin_cuda(x, self.dim).to(x.device)