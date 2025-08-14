import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

smooth_l1_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_loss_kernel(
    const float* predictions, const float* targets, float* loss, int size, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabs(diff);
        if (abs_diff <= beta) {
            loss[idx] = 0.5 * diff * diff;
        } else {
            loss[idx] = beta * (abs_diff - 0.5 * beta);
        }
    }
}

torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets, float beta) {
    auto size = predictions.numel();
    auto loss = torch::empty_like(predictions);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    smooth_l1_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        loss.data_ptr<float>(),
        size,
        beta
    );

    return loss.mean();
}
"""

smooth_l1_loss_cpp_source = (
    "torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets, float beta);"
)

smooth_l1_loss = load_inline(
    name="smooth_l1_loss",
    cpp_sources=smooth_l1_loss_cpp_source,
    cuda_sources=smooth_l1_loss_source,
    functions=["smooth_l1_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = 1.0  # Default value as per PyTorch's smooth_l1_loss
        self.smooth_l1_loss_cuda = smooth_l1_loss

    def forward(self, predictions, targets):
        return self.smooth_l1_loss_cuda.smooth_l1_loss_cuda(predictions, targets, self.beta)