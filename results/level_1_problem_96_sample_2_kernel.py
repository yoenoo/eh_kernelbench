import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

smooth_l1_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void smooth_l1_loss_kernel(const float* predictions, const float* targets, float* loss, int batch_size, int dim) {
    extern __shared__ float shared[];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float total_loss = 0.0;

    while (index < batch_size * dim) {
        int x = index % dim;
        int b = index / dim;
        float diff = predictions[index] - targets[index];
        float abs_diff = fabs(diff);
        float quad = abs_diff < 1.0 ? 0.5 * diff * diff : abs_diff - 0.5f;
        total_loss += quad;
        index += stride;
    }

    // Use shared memory for block reduction
    shared[threadIdx.x] = total_loss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(loss, shared[0]);
    }
}

torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int batch_size = predictions.size(0);
    int dim = predictions.numel() / batch_size;
    float* loss_data;
    cudaMalloc((void**)&loss_data, sizeof(float));
    cudaMemset(loss_data, 0, sizeof(float));

    int block_size = 256;
    int grid_size = (batch_size * dim + block_size - 1) / block_size;

    smooth_l1_loss_kernel<<<grid_size, block_size, block_size * sizeof(float), 0>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        loss_data,
        batch_size,
        dim
    );

    torch::Tensor total_loss = torch::zeros(1, predictions.device());
    cudaMemcpy(total_loss.data_ptr<float>(), loss_data, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(loss_data);

    return total_loss / (batch_size * dim);
}
"""

smooth_l1_loss_cpp = "torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the custom CUDA kernel
smooth_l1_loss = load_inline(
    name="smooth_l1_loss",
    cpp_sources=smooth_l1_loss_cpp,
    cuda_sources=smooth_l1_loss_source,
    functions=["smooth_l1_loss_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.smooth_l1_loss_cuda = smooth_l1_loss

    def forward(self, predictions, targets):
        return self.smooth_l1_loss_cuda.smooth_l1_loss_cuda(predictions, targets)