import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

smooth_l1_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>

template <typename scalar_t>
__global__ void smooth_l1_loss_forward_kernel(const scalar_t* predictions, const scalar_t* targets, scalar_t* output, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        scalar_t diff = predictions[idx] - targets[idx];
        scalar_t abs_diff = fabs(diff);
        if (abs_diff <= 1) {
            output[idx] = 0.5 * diff * diff;
        } else {
            output[idx] = abs_diff - 0.5;
        }
    }
}

at::Tensor smooth_l1_loss_cuda_forward(at::Tensor predictions, at::Tensor targets) {
    int count = predictions.numel();
    auto output = at::empty_like(predictions);
    int block_size = 256;
    int num_blocks = (count + block_size - 1) / block_size;
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(predictions.type(), "smooth_l1_loss_forward", ([&] {
        smooth_l1_loss_forward_kernel<scalar_t><<<num_blocks, block_size, 0, stream>>>(
            predictions.data<scalar_t>(),
            targets.data<scalar_t>(),
            output.data<scalar_t>(),
            count);
    }));

    return output.mean();
}

"""

smooth_l1_loss_cpp_source = "at::Tensor smooth_l1_loss_cuda_forward(at::Tensor predictions, at::Tensor targets);"

smooth_l1_loss_cuda = load_inline(
    name="smooth_l1_loss_cuda",
    cpp_sources=smooth_l1_loss_cpp_source,
    cuda_sources=smooth_l1_loss_source,
    functions=["smooth_l1_loss_cuda_forward"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.forward_op = smooth_l1_loss_cuda.smooth_l1_loss_cuda_forward

    def forward(self, predictions, targets):
        return self.forward_op(predictions, targets)