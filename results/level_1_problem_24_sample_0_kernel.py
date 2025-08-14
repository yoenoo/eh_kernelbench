import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

logsoftmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void logsoftmax_forward_kernel(const float* input, float* output, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int feature_idx = threadIdx.x;

    if (feature_idx >= dim) return;

    float max_val = -FLT_MAX;
    // Find the maximum value for each batch
    for (int i = 0; i < dim; i++) {
        float val = input[batch_idx * dim + i];
        if (val > max_val) {
            max_val = val;
        }
    }

    float sum_exp = 0.0;
    // Compute the sum of exp(x_i - max)
    for (int i = 0; i < dim; i++) {
        sum_exp += exp(input[batch_idx * dim + i] - max_val);
    }

    // Compute logsoftmax
    output[batch_idx * dim + feature_idx] = input[batch_idx * dim + feature_idx] - max_val - log(sum_exp);
}

torch::Tensor logsoftmax_forward_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int dim = input.size(1);

    auto output = torch::empty_like(input);

    dim3 threads(dim);
    dim3 blocks(batch_size);

    logsoftmax_forward_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);

    return output;
}
"""

logsoftmax_cpp_source = (
    "torch::Tensor logsoftmax_forward_cuda(torch::Tensor input);"
)

logsoftmax_extension = load_inline(
    name="logsoftmax",
    cpp_sources=logsoftmax_cpp_source,
    cuda_sources=logsoftmax_source,
    functions=["logsoftmax_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.logsoftmax_forward_cuda = logsoftmax_extension.logsoftmax_forward_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logsoftmax_forward_cuda(x)