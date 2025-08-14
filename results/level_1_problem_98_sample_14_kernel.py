import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for KL divergence computation
kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ 
void kl_div_kernel(const float* predictions, const float* targets, float* out, int batch_size, int dim_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float sum = 0.0;
        for (int d = 0; d < dim_size; ++d) {
            int pos = idx * dim_size + d;
            float p = predictions[pos];
            float t = targets[pos];
            if (p > 1e-20) {  // Avoid log(0) which would be -inf
                sum += t * (log(t) - log(p));
            }
        }
        out[idx] = sum / dim_size;  // Reduction over the last dimension
    }
}

torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets) {
    const int batch_size = predictions.size(0);
    const int dim_size = predictions.size(1);

    auto out = torch::empty({batch_size}, predictions.options());

    const int threads_per_block = 256;
    const int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;

    kl_div_kernel<<<blocks_per_grid, threads_per_block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        dim_size
    );

    return out;
}
"""

kl_div_cpp_source = "torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code
kl_div = load_inline(
    name="kl_div",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=["kl_div_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_div_cuda_op = kl_div

    def forward(self, predictions, targets):
        # Ensure the inputs are on the same device (probably CUDA)
        predictions = predictions.cuda()
        targets = targets.cuda()
        # Compute KL divergence using custom kernel
        return self.kl_div_cuda_op.kl_div_cuda(predictions, targets).mean()
        # Note: The original uses 'batchmean' reduction, which averages over batch and features.
        # However, the current kernel averages over the last dimension (features) and then takes
        # mean over the batch. Alternative: adjust the kernel to compute total and divide by batch*dim.
        # Alternatively, adjust the kernel to accumulate total per sample and then take mean.
        # For simplicity, here we take the mean of the per-sample averages to maintain consistency.