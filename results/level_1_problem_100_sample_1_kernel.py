import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the Hinge Loss computation
hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* out, int batch_size) {
    __shared__ float block_sum;
    // Initialize shared memory for block reduction
    if (threadIdx.x == 0) {
        block_sum = 0.0f;
    }
    __syncthreads();

    // Each thread computes the clamp value for its element
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val = 0.0f;
    if (idx < batch_size) {
        float prod = predictions[idx] * targets[idx];
        val = (1 - prod) > 0 ? (1 - prod) : 0.0f;
    }

    // Use parallel reduction to sum all values in the block
    __shared__ float shared_data[1024];
    shared_data[threadIdx.x] = val;
    __syncthreads();

    for (int s = 1; s <= blockDim.x; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out, shared_data[0]);
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.numel();
    auto out = torch::zeros(1, predictions.options());

    const int block_size = 1024;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    hinge_loss_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), out.data_ptr<float>(), batch_size);

    return out / batch_size;
}
"""

hinge_loss_cpp_source = "torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code for Hinge Loss
hinge_loss = load_inline(
    name="hinge_loss",
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
        self.hinge_loss = hinge_loss

    def forward(self, predictions, targets):
        return self.hinge_loss.hinge_loss_cuda(predictions, targets)