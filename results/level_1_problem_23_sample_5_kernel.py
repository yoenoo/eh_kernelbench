import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_kernel(const float* input, float* output, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int feature_idx = threadIdx.x;

    // Ensures that all threads in the block have the same view of the input
    __shared__ float max_value_buf[256];
    if (feature_idx < dim) {
        float val = input[batch_idx * dim + feature_idx];
        max_value_buf[feature_idx] = val;
    }
    __syncthreads();

    // Compute the maximum value for the current batch item
    float max_val = -INFINITY;
    for (int idx = 0; idx < dim; idx++) {
        if (threadIdx.x == 0 && idx < dim) {
            if (max_value_buf[idx] > max_val) {
                max_val = max_value_buf[idx];
            }
        }
        __syncthreads();
    }

    // Subtract max to prevent overflow
    if (feature_idx < dim) {
        float exp_val = exp(input[batch_idx * dim + feature_idx] - max_val);
        max_value_buf[feature_idx] = exp_val;
    }
    __synsthreads();

    // Compute the sum of exponentials
    float sum = 0.0;
    for (int idx = 0; idx < dim; idx++) {
        if (threadIdx.x == 0 && idx < dim) {
            sum += max_value_buf[idx];
        }
        __syncthreads();
    }

    // Compute softmax
    if (feature_idx < dim) {
        float result = max_value_buf[feature_idx] / sum;
        output[batch_idx * dim + feature_idx] = result;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int dim = input.size(1);

    auto output = torch::empty_like(input);

    dim3 threadsPerBlock(dim); // Each feature is handled by a thread
    dim3 numBlocks(batch_size); // Each batch item processed in a separate block

    softmax_kernel<<<numBlocks, threadsPerBlock>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );

    return output;
}
"""

softmax_cuda_header = "torch::Tensor softmax_cuda(torch::Tensor input);"

softmax = load_inline(
    name="custom_softmax",
    cpp_sources=softmax_cuda_header,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax.softmax_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x.cuda()]

def get_init_inputs():
    return []