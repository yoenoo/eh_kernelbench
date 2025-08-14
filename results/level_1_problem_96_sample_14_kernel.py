import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Smooth L1 Loss
smooth_l1_loss_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_loss_kernel(const float* predictions, const float* targets, float* output, int count) {
    extern __shared__ float shared[];
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float sum = 0.0;

    // Thread-private accumulation
    for (int i = idx; i < count; i += blockDim.x * 2 * gridDim.x) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabs(diff);
        float loss = (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
        sum += loss;
    }

    // Thread block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __shared__ float shared_sum[1024]; // Adjust size based on max block size
        __syncthreads();
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
    }
    if (threadIdx.x == 0) {
        atomicAdd(output, shared[0]);
    }
}

torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int count = predictions.numel();
    int block_size = 256;
    int grid_size = (count + block_size * 2 - 1) / (block_size * 2);

    float* output = new float[1];
    *output = 0.0f;
    cudaMallocManaged(&output, sizeof(float));

    smooth_l1_loss_kernel<<<grid_size, block_size, sizeof(float)*block_size>>>(
        predictions.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        output,
        count
    );

    cudaDeviceSynchronize();
    float result = *output / count;
    cudaFree(output);
    return torch::tensor(result, predictions.options());
}
"""

# Compile the custom CUDA kernel
smooth_l1_loss = load_inline(
    name='smooth_l1_loss',
    cuda_sources=[smooth_l1_loss_source],
    functions=['smooth_l1_loss_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.smooth_l1_loss_cuda = smooth_l1_loss

    def forward(self, predictions, targets):
        return self.smooth_l1_loss_cuda.smooth_l1_loss_cuda(predictions, targets)