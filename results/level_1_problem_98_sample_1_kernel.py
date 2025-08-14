import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void kl_div_forward_kernel(const scalar_t* __restrict__ pred_log, 
                                     const scalar_t* __restrict__ target, 
                                     scalar_t* out, 
                                     const int batch_size,
                                     const int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        scalar_t sum = 0;
        for (int d = 0; d < dim; ++d) {
            int offset = idx * dim + d;
            sum += target[offset] * (log(target[offset] + 1e-20) - pred_log[offset]);
        }
        out[idx] = sum;
    }
}

torch::Tensor kl_div_forward_cuda(torch::Tensor pred_log, torch::Tensor target) {
    const int batch_size = pred_log.size(0);
    const int dim = pred_log.size(1);

    auto out = torch::empty({batch_size}, pred_log.options());

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(pred_log.type(), "kl_div_forward_cuda", ([&] {
        kl_div_forward_kernel<scalar_t><<<num_blocks, block_size>>>(
            pred_log.data_ptr<scalar_t>(),
            target.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            dim);
    }));

    // Compute mean over batch and elements
    auto sum = out.sum();
    return sum / (batch_size * dim);
}
"""

kl_div_cpp_source = (
    "torch::Tensor kl_div_forward_cuda(torch::Tensor pred_log, torch::Tensor target);"
)

# Compile the inline CUDA code for KL divergence
kl_div = load_inline(
    name="kl_div_forward",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=["kl_div_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.kl_div = kl_div

    def forward(self, predictions, targets):
        # Compute log(predictions) in fused way to avoid separate log operation
        pred_log = torch.log(predictions + 1e-20)  # Add small epsilon to prevent log(0)
        return self.kl_div.kl_div_forward_cuda(pred_log, targets)