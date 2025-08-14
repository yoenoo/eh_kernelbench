import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the GELU activation function
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_forward(const float* x, float* out, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        const float sqrt_2_over_pi = 0.7978845608f;
        const float bias = 0.044715f;
        float tanh_out = tanh(sqrt_2_over_pi * (xi + bias * xi * xi * xi));
        out[idx] = 0.5f * xi * (1.0f + tanh_out);
    }
}

__global__ void gelu_backward(const float* x, const float* grad_out, float* grad_in, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        const float sqrt_a = 0.7978845608f;  // sqrt(2/pi)
        const float a = 0.044715f;
        const float tanhcoeff = 0.79788456f * 0.33267f;  // sqrt(2/pi) * (1/pi^3)^{1/4}
        float r = xi * 0.3989423f;  // 1/sqrt(2pi)
        float r2 = r * r;
        float x_pow3 = xi * xi * xi;
        float tanh_out = tanh(tanhcoeff * (x_pow3 + 1.702f * xi)));
        float f = 0.5f * (1.0f + tanh_out);
        float grad = grad_out[idx];
        grad_in[idx] = grad * (f + xi * (0.5f + a * (x_pow3 + xi * 3.0f * 0.044715f * xi * xi)));
    }
}

torch::Tensor gelu_forward_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    const int size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_forward<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

std::tuple<torch::Tensor> gelu_backward_cuda(torch::Tensor x, torch::Tensor grad_out) {
    auto grad_in = torch::empty_like(x);
    const int size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_backward<<<num_blocks, block_size>>>(x.data_ptr<float>(), grad_out.data_ptr<float>(), grad_in.data_ptr<float>(), size);
    return grad_in;
}
"""

gelu_cpp_source = """
torch::Tensor gelu_forward_cuda(torch::Tensor x);
std::tuple<torch::Tensor> gelu_backward_cuda(torch::Tensor x, torch::Tensor grad_out);
"""

gelu_cuda = load_inline(
    name="gelu_cuda",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_forward_cuda", "gelu_backward_cuda"],
    verbose=True,
    extra_cflags=["-g", "-w"],
    extra_ldflags=["-g"]
)

class ModelNew(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return gelu_cuda.gelu_forward_cuda(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = gelu_cuda.gelu_backward_cuda(x, grad_output)
        return grad_input

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return ModelNew.apply(x)

# The original get_inputs and get_init_inputs remain unchanged