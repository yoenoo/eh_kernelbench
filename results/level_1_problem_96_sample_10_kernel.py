import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

smooth_l1_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void smooth_l1_loss_kernel(const float* predictions, const float* targets, float* loss, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabs(diff);
        if (abs_diff < 1) {
            loss[idx] = 0.5 * diff * diff;
        } else {
            loss[idx] = abs_diff - 0.5;
        }
    }
}

torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int batch_size = predictions.size(0);
    int dim = predictions.size(1);

    auto loss = torch::zeros_like(predictions);

    const int block_size = 256;
    int num_elements = batch_size * dim;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    smooth_l1_loss_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), loss.data_ptr<float>(), batch_size, dim);

    // Sum over the dimension and then take the mean over the batch
    auto loss_summed = loss.sum({1});
    auto loss_mean = loss_sumed.mean();
    return loss_mean;
}
"""

smooth_l1_loss_cpp_source = "torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code
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
        self.smooth_l1_loss = smooth_l1_loss

    def forward(self, predictions, targets):
        return self.smooth_l1_loss.smooth_l1_loss_cuda(predictions, targets)