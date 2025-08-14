import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l2norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void l2norm_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, int batch_size, int dim) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch < batch_size) {
        scalar_t norm = 0;
        for (int d = 0; d < dim; ++d) {
            scalar_t val = x[batch * dim + d];
            norm += val * val;
        }
        norm = sqrt(norm);
        if (norm == 0) norm = 1;  // Avoid division by zero
        for (int d = 0; d < dim; ++d) {
            y[batch * dim + d] = x[batch * dim + d] / norm;
        }
    }
}

torch::Tensor l2norm_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);

    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int grid_size = (batch_size + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "l2norm_cuda", ([&] {
        l2norm_kernel<scalar_t><<<grid_size, block_size>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            batch_size,
            dim
        );
    }));

    return y;
}
"""

l2norm_cpp_source = "torch::Tensor l2norm_cuda(torch::Tensor x);"

l2norm = load_inline(
    name="l2norm",
    cpp_sources=l2norm_cpp_source,
    cuda_sources=l2norm_source,
    functions=["l2norm_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l2norm = l2norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2norm.l2norm_cuda(x)

def get_inputs():
    # Ensure tensors are on GPU for CUDA kernel
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []