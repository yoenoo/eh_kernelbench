import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized Cross Entropy Loss computation
# The implementation combines log_softmax and nll_loss into a single kernel
cross_entropy_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void cross_entropy_kernel(const float* predictions, const int64_t* targets, float* output, int batch_size, int num_classes) {
    extern __shared__ float block_max[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid >= batch_size)
        return;

    // Compute log_softmax in place
    float max_val = -INFINITY;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        if (predictions[bid * num_classes + i] > max_val)
            max_val = predictions[bid * num_classes + i];
    }

    // Block reduction to find the maximum value
    __shared__ float shared_max[32]; // Assuming block size <= 1024 (32 elements for 1024 threads)
    shared_max[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    max_val = shared_max[0];
    __syncthreads();

    float sum_exp = 0.0;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        sum_exp += exp(predictions[bid * num_classes + i] - max_val);
    }

    __syncthreads();

    // Block-wise sum reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, s);
        }
        __syncthreads();
    }

    if (tid == 0) {
        float log_softmax_val = predictions[bid * num_classes + targets[bid]] - max_val - log(sum_exp);
        output[bid] = -log_softmax_val;
    }
    __syncthreads();
}

torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    auto output = torch::empty({batch_size}, predictions.options());

    int block_size = 256; // Tune block size for best performance
    int shared_mem = sizeof(float) * ((block_size <= 1024) ? 32 : 64); // Shared memory for block_max

    cross_entropy_kernel<<<batch_size, block_size, shared_mem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        batch_size,
        num_classes
    );

    return output.mean();
}
"""

cross_entropy_cpp_source = "torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the custom CUDA kernel
cross_entropy = load_inline(
    name="cross_entropy",
    cpp_sources=cross_entropy_cpp_source,
    cuda_sources=cross_entropy_source,
    functions=["cross_entropy_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy = cross_entropy

    def forward(self, predictions, targets):
        return self.cross_entropy.cross_entropy_cuda(predictions, targets)