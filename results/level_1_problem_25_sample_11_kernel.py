import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_forward_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        out[idx] = xi / (1.0f + expf(-xi));
    }
}

__global__ void swish_backward_kernel(const float* x, const float* grad_out, float* grad_x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float sigmoid_xi = 1.0f / (1.0f + expf(-xi));
        float grad = grad_out[idx] * (sigmoid_xi * (1.0f + xi * (1.0f - sigmoid_xi)));
        grad_x[idx] = grad;
    }
}

torch::Tensor swish_forward_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    auto size = x.numel();
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_forward_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size
    );
    return out;
}

torch::Tensor swish_backward_cuda(torch::Tensor x, torch::Tensor grad_out) {
    auto grad_x = torch::empty_like(x);
    auto size = x.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_backward_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), grad_out.data_ptr<float>(), grad_x.data_ptr<float>(), size
    );
    return grad_x;
}
"""

swish_cpp_source = """
torch::Tensor swish_forward_cuda(torch::Tensor x);
torch::Tensor swish_backward_cuda(torch::Tensor x, torch::Tensor grad_out);
"""

# Compile the inline CUDA code for Swish
swish_ops = load_inline(
    name="swish_ops",
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_source,
    functions=["swish_forward_cuda", "swish_backward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.swish_forward = swish_ops
        self.swish_backward = swish_ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish_forward.swish_forward_cuda(x)

    def backward(self, x, grad_out):
        return self.swish_backward.swish_backward_cuda(x, grad_out)