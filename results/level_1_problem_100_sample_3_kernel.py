import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for fused Hinge Loss computation
hinge_loss_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_hinge_loss_kernel(
    const float* predictions, const float* targets, float* output, int n) {
    extern __shared__ float shared[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;

    if (idx < n) {
        float prod = predictions[idx] * targets[idx];
        float val = 1.0 - prod;
        sum = (val > 0) ? val : 0.0;
    } else {
        sum = 0.0;
    }

    // Use shared memory for block-wise reduction
    int tid = threadIdx.x;
    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

torch::Tensor fused_hinge_loss(
    torch::Tensor predictions, torch::Tensor targets) {
    const int n = predictions.numel();
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    float* output_data;
    cudaMalloc(&output_data, sizeof(float));
    cudaMemset(output_data, 0, sizeof(float));

    fused_hinge_loss_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(), targets.data_ptr<float>(), output_data, n);

    float result;
    cudaMemcpy(&result, output_data, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(output_data);

    return torch::tensor(result / n, predictions.options());
}
"""

# Header declarations for compilation
hinge_loss_header = "torch::Tensor fused_hinge_loss(torch::Tensor predictions, torch::Tensor targets);"

# Load the CUDA kernel
fused_hinge_loss = load_inline(
    name="fused_hinge_loss",
    cpp_sources=hinge_loss_header,
    cuda_sources=hinge_loss_kernel,
    functions=["fused_hinge_loss"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.fused_hinge_loss = fused_hinge_loss

    def forward(self, predictions, targets):
        return self.fused_hinge_loss.fused_hinge_loss(predictions, targets)