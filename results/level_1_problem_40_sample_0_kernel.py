cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class CustomLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, eps=1e-5):
        # Save necessary variables for backward pass
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, unbiased=False, keepdim=True)
        std = torch.sqrt(var + eps)
        x_normalized = (input - mean) / std
        output = x_normalized * weight + bias if weight is not None and bias is not None else x_normalized
        ctx.save_for_backward(input, weight, bias, mean, var, std)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mean, var, std = ctx.saved_tensors
        eps = ctx.eps
        N = input.size(-1)

        # Compute gradients for inputs
        dx_normalized = grad_output * weight if weight is not None else grad_output
        dvar = torch.sum(dx_normalized * (input - mean) * (-0.5) * torch.pow(var + eps, -1.5), dim=-1, keepdim=True)
        dmean = torch.sum(dx_normalized * (-1.0) / std, dim=-1, keepdim=True) + dvar * torch.sum(-2.0 * (input - mean), dim=-1, keepdim=True) / N
        dx = dx_normalized / std + dvar * 2.0 * (input - mean) / N + dmean / N

        # Compute gradients for weight and bias
        dweight = torch.sum(grad_output * (input - mean) / torch.sqrt(var + eps), dim=(-2, -1), keepdim=True) if weight is not None else None
        dbias = torch.sum(grad_output, dim=(-2, -1), keepdim=True) if bias is not None else None

        return dx, dweight, dbias, None

class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(CustomLayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(*self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        return CustomLayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = CustomLayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)