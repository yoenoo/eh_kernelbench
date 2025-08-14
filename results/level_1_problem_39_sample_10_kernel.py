import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l2_norm_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void l2_norm_kernel(const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> x,
                              torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> output,
                              const int batch_size, const int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        scalar_t norm = 0;
        for (int d = 0; d < dim; ++d) {
            norm += x[idx][d] * x[idx][d];
        }
        norm = sqrt(norm);
        if (norm > 1e-12) {
            for (int d = 0; d < dim; ++d) {
                output[idx][d] = x[idx][d] / norm;
            }
        } else {
            // Avoid division by zero by setting to zero
            for (int d = 0; d < dim; ++d) {
                output[idx][d] = 0;
            }
        }
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor x) {
    const auto batch_size = x.size(0);
    const auto dim = x.size(1);

    auto output = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "l2_norm_cuda", ([&] {
        l2_norm_kernel<scalar_t><<<num_blocks, block_size>>>(
            x.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            batch_size, dim);
    }));

    return output;
}
"""

l2_norm_cpp_source = "torch::Tensor l2_norm_cuda(torch::Tensor x);"

l2_norm = load_inline(
    name="l2_norm",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_kernel_source,
    functions=["l2_norm_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l2_norm_cuda_op = l2_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2_norm_cuda_op.l2_norm_cuda(x)