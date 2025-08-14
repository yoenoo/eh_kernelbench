import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Hinge Loss computation
hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ 
void hinge_loss_kernel(const float* predictions, const float* targets, float* output, int batch_size) {
    extern __shared__ volatile float shared_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory for reduction
    float sum = 0.0;
    if (idx < batch_size) {
        float pred = predictions[idx];
        float target = targets[idx];
        float loss_val = 1.0 - pred * target;
        sum = (loss_val > 0.0) ? loss_val : 0.0;
    }

    // Parallel reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {
        __shared__ float shared_sum[256]; // max thread block size
        if (tid % (2 * s) == 0) {
            sum += shared_data[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output, sum);
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int batch_size = predictions.size(0);
    auto output = torch::zeros(1, predictions.options());

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    dim3 blocks(num_blocks);
    dim3 threads(block_size);
    int shared_mem = sizeof(float) * block_size;

    hinge_loss_kernel<<<blocks, threads, shared_mem>>>(
        predictions.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size
    );

    // Compute mean
    output = output / batch_size;
    return output;
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
    extra_cflags=["-D_FORCE_INLINES"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hinge_loss = hinge_loss

    def forward(self, predictions, targets):
        return self.hinge_loss.hinge_loss_cuda(predictions, targets)