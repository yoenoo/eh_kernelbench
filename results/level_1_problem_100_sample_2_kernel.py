import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* out, int batch_size) {
    __shared__ float block_sum;
    // Initialize block sum
    if (threadIdx.x == 0) {
        block_sum = 0.0f;
    }
    __syncthreads();

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < batch_size) {
        float val = 1.0f - predictions[idx] * targets[idx];
        val = (val > 0.0f) ? val : 0.0f;
        atomicAdd(&block_sum, val);
    }
    __syncthreads();

    // Write the block result
    if (threadIdx.x == 0) {
        atomicAdd(out, block_sum);
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.numel();
    auto out = torch::empty(1, dtype(predictions.dtype()), device(predictions.device()));

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    hinge_loss_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(),
                                                 targets.data_ptr<float>(),
                                                 out.data_ptr<float>(),
                                                 batch_size);

    return out / batch_size;
}
"""

hinge_loss_cpp_source = "torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

hinge_loss_ext = load_inline(
    name="hinge_loss_cuda",
    cpp_sources=hinge_loss_cpp_source,
    cuda_sources=hinge_loss_source,
    functions=["hinge_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hinge_loss_cuda = hinge_loss_ext

    def forward(self, predictions, targets):
        return self.hinge_loss_cuda.hinge_loss_cuda(predictions, targets)