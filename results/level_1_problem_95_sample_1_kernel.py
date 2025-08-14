import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cross_entropy_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/ATen.h>

template <typename scalar_t>
__global__ void cross_entropy_kernel(const scalar_t* predictions, 
                                    const int64_t* targets,
                                    scalar_t* loss,
                                    int batch_size,
                                    int num_classes) {
    extern __shared__ scalar_t shared[];

    int tid = threadIdx.x;
    scalar_t sum = 0.0;
    scalar_t prob;
    scalar_t loss_val = 0.0;

    // Load prediction for current thread
    scalar_t pred = predictions[blockIdx.x * num_classes + tid];

    // Compute max for stability
    __shared__ scalar_t block_max;
    if (tid == 0) {
        block_max = std::numeric_limits<scalar_t>::lowest();
        for (int i = 0; i < num_classes; i++) {
            scalar_t val = predictions[blockIdx.x * num_classes + i];
            if (val > block_max) block_max = val;
        }
    }
    __syncthreads();

    // Compute log_softmax
    pred -= block_max;
    prob = exp(pred);
    sum += prob;
    __shared__ scalar_t total_sum;
    if (tid == 0) {
        total_sum = 0;
        for (int i = 0; i < num_classes; i++) {
            scalar_t val = exp(predictions[blockIdx.x * num_classes + i] - block_max);
            total_sum += val;
        }
        prob = exp(predictions[blockIdx.x * num_classes + targets[blockIdx.x]] - block_max);
        loss_val = -log(prob / total_sum);
    }
    __syncthreads();

    if (tid == 0) {
        loss[blockIdx.x] = loss_val;
    }
}

torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets) {
    const int batch_size = predictions.size(0);
    const int num_classes = predictions.size(1);

    auto loss = torch::zeros({batch_size}, predictions.options());

    const int block_size = 256;
    const dim3 grid(batch_size);
    const dim3 block(num_classes > 256 ? 256 : num_classes); // Limit block size to 256

    cross_entropy_kernel<float><<<grid, block, block.x * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        loss.data_ptr<float>(),
        batch_size,
        num_classes
    );

    // Compute mean loss
    auto mean_loss = loss.mean();

    return mean_loss;
}
"""

cross_entropy_cpp_source = "torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets);"

cross_entropy = load_inline(
    name="cross_entropy",
    cpp_sources=cross_entropy_cpp_source,
    cuda_sources=cross_entropy_source,
    functions=["cross_entropy_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = cross_entropy

    def forward(self, predictions, targets):
        return self.cross_entropy.cross_entropy_cuda(predictions, targets)