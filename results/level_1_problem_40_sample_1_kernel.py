import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class CustomLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, eps):
        # Save necessary variables for backward
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, unbiased=False, keepdim=True)
        std = torch.sqrt(var + eps)
        x_norm = (input - mean) / std
        output = x_norm * weight + bias if weight is not None else x_norm
        ctx.save_for_backward(x_norm, weight, std)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_norm, weight, std = ctx.saved_tensors
        N = x_norm.size(-1)
        if N == 0:
            return None, None, None
        grad_input = grad_weight = grad_bias = None

        if weight is not None:
            grad_weight = torch.sum(grad_output * x_norm, dim=-1)
            grad_bias = grad_output.sum(dim=-1)

            grad_input = (grad_output - grad_weight / N * x_norm - grad_bias / N) * weight / std
        else:
            grad_input = grad_output * 1.0 / std
            grad_weight = None
            grad_bias = grad_output.sum(dim=-1).sum(dim=-1) if grad_output.dim() > 1 else grad_output.sum()

        grad_input = grad_input * 1.0  # Broadcast to match input dimension
        return grad_input, grad_weight, grad_bias, None

class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.randn(normalized_shape))
        self.bias = nn.Parameter(torch.randn(normalized_shape))
        self.eps = 1e-5  # Using default epsilon

    def forward(self, x):
        return CustomLayerNorm.apply(x, self.weight, self.bias, self.eps)

def get_inputs():
    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 256
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [(64, 256, 256)]