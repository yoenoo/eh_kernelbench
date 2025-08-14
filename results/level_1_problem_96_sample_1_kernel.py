import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

smooth_l1_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void smooth_l1_loss_forward_kernel(const scalar_t* __restrict__ predictions, const scalar_t* __restrict__ targets, scalar_t* __restrict__ output, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        scalar_t diff = predictions[idx] - targets[idx];
        scalar_t abs_diff = fabs(diff);
        if (abs_diff <= 1) {
            output[idx] = 0.5 * diff * diff;
        } else {
            output[idx] = abs_diff - 0.5;
        }
    }
}

at::Tensor smooth_l1_loss_cuda(at::Tensor predictions, at::Tensor targets) {
    auto n_elements = predictions.numel();
    auto output = at::empty({n_elements}, predictions.options());
    
    const int block_size = 256;
    const int num_blocks = (n_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(predictions.type(), "smooth_l1_loss_forward", ([&] {
        smooth_l1_loss_forward_kernel<scalar_t><<<num_blocks, block_size>>>(
            predictions.data<scalar_t>(),
            targets.data<scalar_t>(),
            output.data<scalar_t>(),
            n_elements);
    }));

    auto mean_output = output.mean();
    return mean_output;
}
"""

smooth_l1_loss_header = """
#include <torch/extension.h>
"""

smooth_l1_loss = load_inline(
    name="smooth_l1_loss_cuda",
    cpp_sources=smooth_l1_loss_header,
    cuda_sources=smooth_l1_loss_source,
    functions=["smooth_l1_loss_cuda"],
    verbose=True,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_smooth_l1_loss = smooth_l1_loss

    def forward(self, predictions, targets):
        return self.custom_smooth_l1_loss.smooth_l1_loss_cuda(predictions, targets)