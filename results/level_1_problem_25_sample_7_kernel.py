import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_forward_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        y[idx] = xi / (1.0f + exp(-xi));
    }
}

__global__ void swish_backward_kernel(const float* x, const float* grad_y, float* grad_x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float sigmoid_xi = 1.0f / (1.0f + exp(-xi));
        grad_x[idx] = grad_y[idx] * (sigmoid_xi * (1.0f + xi * (1.0f - sigmoid_xi)));
    }
}

torch::Tensor swish_forward_cuda(torch::Tensor x) {
    auto output = torch::empty_like(x);
    int size = x.numel();

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    swish_forward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}

torch::Tensor swish_backward_cuda(torch::Tensor x, torch::Tensor grad_y) {
    auto grad_x = torch::empty_like(x);
    int size = x.numel();

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    swish_backward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), grad_y.data_ptr<float>(), grad_x.data_ptr<float>(), size);

    return grad_x;
}
"""

# Compile the inline CUDA code for Swish
swish_extension = load_inline(
    name="swish_cuda",
    cpp_sources="",
    cuda_sources=swish_source,
    functions=["swish_forward_cuda", "swish_backward_cuda"],
    verbose=True,
    with_cuda=True
)

class SwishAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_extension.swish_forward_cuda(x)

    @staticmethod
    def backward(ctx, grad_y):
        x, = ctx.saved_tensors
        return swish_extension.swish_backward_cuda(x, grad_y)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SwishAutogradFunction.apply(x)