import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_forward_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        y[idx] = xi / (1.0f + expf(-xi));
    }
}

__global__ void swish_backward_kernel(const float* x, const float* grad_y, float* grad_x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float sigmoid_xi = 1.0f / (1.0f + expf(-xi));
        grad_x[idx] = grad_y[idx] * (sigmoid_xi * (1.0f + xi * (1.0f - sigmoid_xi)));
    }
}

torch::Tensor swish_forward_cuda(torch::Tensor x) {
    auto output = torch::empty_like(x);
    const int size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    swish_forward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}

torch::Tensor swish_backward_cuda(torch::Tensor x, torch::Tensor grad_y) {
    auto grad_x = torch::empty_like(x);
    const int size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    swish_backward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), grad_y.data_ptr<float>(), grad_x.data_ptr<float>(), size);
    return grad_x;
}
"""

swish_cpp_source = """
torch::Tensor swish_forward_cuda(torch::Tensor x);
torch::Tensor swish_backward_cuda(torch::Tensor x, torch::Tensor grad_y);
"""

swish_ext = load_inline(
    name="swish_ops",
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_source,
    functions=["swish_forward_cuda", "swish_backward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.swish_forward = swish_ext.swish_forward_cuda
        self.swish_backward = swish_ext.swish_backward_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish_forward(x)

    def backward(self, x: torch.Tensor, grad_y: torch.Tensor) -> torch.Tensor:
        return self.swish_backward(x, grad_y)