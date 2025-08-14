import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_forward(const float* input, float* output, const float min_val, const float max_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
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

__global__ void hardtanh_backward(const float* grad_output, const float* input, float* grad_input, const float min_val, const float max_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        grad_input[idx] = (val > min_val && val < max_val) ? grad_output[idx] : 0.0;
    }
}

torch::Tensor hardtanh_forward_cuda(torch::Tensor input, float min_val, float max_val) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardtanh_forward<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), min_val, max_val, size);
    return output;
}

torch::Tensor hardtanh_backward_cuda(torch::Tensor grad_output, torch::Tensor input, float min_val, float max_val) {
    auto grad_input = torch::empty_like(input);
    const int size = input.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardtanh_backward<<<num_blocks, block_size>>>(grad_output.data_ptr<float>(), input.data_ptr<float>(), grad_input.data_ptr<float>(), min_val, max_val, size);
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hardtanh_forward_cuda, "Hardtanh forward");
    m.def("backward", &hardtanh_backward_cuda, "Hardtanh backward");
}
"""

hardtanh_cpp_source = """
#include <torch/extension.h>
torch::Tensor forward(torch::Tensor input, float min_val, float max_val);
torch::Tensor backward(torch::Tensor grad_output, torch::Tensor input, float min_val, float max_val);
"""

hardtanh_ops = load_inline(
    name="hardtanh_ops",
    cpp_sources=[hardtanh_cpp_source],
    cuda_sources=[hardtanh_source],
    functions=["forward", "backward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.min_val = -1.0
        self.max_val = 1.0
        self.hardtanh = hardtanh_ops

    def forward(self, x):
        return self.hardtanh.forward(x, self.min_val, self.max_val)

    def backward(self, grad_output, input):
        return self.hardtanh.backward(grad_output, input, self.min_val, self.max_val)

    # Monkey-patch the backward to allow autograd to use our custom kernel
    def hardtanh_backward(self, grad_output, input):
        return self.backward(grad_output, input)

    def __prepare_a_patch(self):
        # Create a dummy Tensor that keeps a reference to this module
        self._backward_proxy = torch.empty(0, requires_grad=True, device=next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else 'cpu')

        # Override the backward hook
        def custom_backward(grad_output):
            input = self._inputs.pop(0)
            return self.hardtanh_backward(grad_output, input)

        self._custom_backward = custom_backward
        return self._backward_proxy

    # Intercept the first forward pass to store input for backward
    def forward(self, x):
        self._inputs = [x.detach().clone()]  # Store input for backward
        output = super().forward(x)
        # Attach the backward hook
        return output.detach() + self.__prepare_a_patch()

    # Need to override parameters to avoid issues with nn.Module's parameter checking
    def parameters(self, recurse=True):
        return []

# Overriding the autograd function to use our custom backward
class HardtanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module):
        ctx.module = module
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        module = ctx.module
        input = module._inputs[0]
        return module.hardtanh_backward(grad_output, input), None

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.min_val = -1.0
        self.max_val = 1.0
        self.hardtanh = hardtanh_ops

    def forward(self, x):
        return HardtanhFunction.apply(x, self)

# Simplified version without autograd hook complexity
class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.min_val = -1.0
        self.max_val = 1.0
        self.hardtanh = hardtanh_ops

    def forward(self, x):
        return self.hardtanh.forward(x, self.min_val, self.max_val)