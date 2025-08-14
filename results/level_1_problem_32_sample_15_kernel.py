import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_forward_kernel(const float* input, float* output, int n, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        if (val < min_val) {
            output[idx] = min_val;
        } else if (val > max_val) {
            output[idx] = max_val;
        } else {
            output[idx] = val;
        }
    }
}

torch::Tensor hardtanh_forward_cuda(torch::Tensor input, float min_val, float max_val) {
    int n = input.numel();
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    hardtanh_forward_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), n, min_val, max_val);
    return output;
}

__global__ void hardtanh_backward_kernel(const float* grad_output, const float* input, float* grad_input, int n, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (input[idx] <= min_val || input[idx] >= max_val) {
            grad_input[idx] = 0;
        } else {
            grad_input[idx] = grad_output[idx];
        }
    }
}

torch::Tensor hardtanh_backward_cuda(torch::Tensor grad_output, torch::Tensor input, float min_val, float max_val) {
    int n = grad_output.numel();
    auto grad_input = torch::zeros_like(grad_output);
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    hardtanh_backward_kernel<<<num_blocks, block_size>>>(grad_output.data_ptr<float>(), input.data_ptr<float>(), grad_input.data_ptr<float>(), n, min_val, max_val);
    return grad_input;
}
"""

hardtanh_cpp_source = """
torch::Tensor hardtanh_forward_cuda(torch::Tensor input, float min_val, float max_val);
torch::Tensor hardtanh_backward_cuda(torch::Tensor grad_output, torch::Tensor input, float min_val, float max_val);
"""

hardtanh_extension = load_inline(
    name="hardtanh_extension",
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_source,
    functions=["hardtanh_forward_cuda", "hardtanh_backward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.min_val = -1.0
        self.max_val = 1.0
        self.forward_op = hardtanh_extension.hardtanh_forward_cuda
        self.backward_op = hardtanh_extension.hardtanh_backward_cuda

    def forward(self, x):
        return self.forward_op(x, self.min_val, self.max_val)

    def backward(self, grad_output, input):
        return self.backward_op(grad_output, input, self.min_val, self.max_val)