import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for KL Divergence
kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void kl_div_kernel(const float* log_probs, const float* targets, float* out, int batch_size, int dim_size) {
    // Compute the KL Divergence for each sample in the batch
    int batch_idx = blockIdx.x;
    float sum = 0.0;
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float log_p = log_probs[batch_idx * dim_size + i];
        float q = targets[batch_idx * dim_size + i];
        if (q > 1e-20) {
            sum += q * (log(1.0 / q) - log_p);
        }
    }
    // Use warp-level reduction for summation
    for (int stride = 128; stride > 0; stride >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, stride, 256);
    }
    if (threadIdx.x == 0) {
        out[batch_idx] = sum / dim_size;
    }
}

torch::Tensor kl_div_cuda(torch::Tensor log_probs, torch::Tensor targets) {
    int batch_size = log_probs.size(0);
    int dim_size = log_probs.size(1);

    auto out = torch::empty({batch_size}, log_probs.options());

    const int block_size = 256;
    const int grid_size = batch_size;

    // Configure kernel with 256 threads per block and 1 block per sample
    kl_div_kernel<<<grid_size, block_size>>>(
        log_probs.data_ptr<float>(),
        targets.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        dim_size
    );

    return out;
}
"""

kl_div_cpp_source = "torch::Tensor kl_div_cuda(torch::Tensor log_probs, torch::Tensor targets);"

# Compile the inline CUDA code for KL Divergence
kl_div = load_inline(
    name="kl_div",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=["kl_div_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.kl_div = kl_div

    def forward(self, predictions, targets):
        # Avoid taking log of zero by clamping predictions
        log_predictions = torch.clamp(predictions, 1e-20, 1.0).log()
        return self.kl_div.kl_div_cuda(log_predictions, targets)