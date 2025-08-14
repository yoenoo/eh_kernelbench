import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class CustomLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, weight, bias, eps=1e-5):
        # Check CUDA availability and move tensors to device
        assert input_tensor.is_cuda, "Input tensor must be on CUDA device"
        assert weight.is_cuda, "Weight tensor must be on CUDA device"
        assert bias.is_cuda, "Bias tensor must be on CUDA device"

        # Dimensions of the input tensor
        batch_size, features, dim1, dim2 = input_tensor.size()
        normalized_shape = (features, dim1, dim2)
        feature_size = features * dim1 * dim2

        # Compute mean and variance
        mean = torch.mean(input_tensor, dim=(1, 2, 3), keepdim=True)
        variance = torch.var(input_tensor, dim=(1, 2, 3), unbiased=False, keepdim=True)

        # Normalize
        x_centered = input_tensor - mean
        std = torch.sqrt(variance + eps)
        x_norm = x_centered / std

        # Apply affine transformation
        output = x_norm * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)

        # Save variables for backward
        ctx.save_for_backward(x_centered, std, weight)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_centered, std, weight = ctx.saved_tensors
        batch_size, features, dim1, dim2 = x_centered.size()
        N = features * dim1 * dim2

        # Compute gradients
        dx_norm = grad_output * weight.view(1, -1, 1, 1)
        dvar = torch.sum(dx_norm * x_centered, dim=(1, 2, 3), keepdim=True) * (-0.5) * (std ** (-3))
        dmu = torch.sum(dx_norm * (-1.0 / std), dim=(1, 2, 3), keepdim=True) + dvar * torch.mean(-2.0 * x_centered, dim=(1, 2, 3), keepdim=True)

        dx = (dx_norm / std) + (dvar * 2 * x_centered / N + dmu / N)

        # Compute gradients for weight and bias
        dweight = torch.sum(grad_output * (x_centered / std), dim=(0, 2, 3), keepdim=True).view(features, -1)
        dbias = torch.sum(grad_output, dim=(0, 2, 3), keepdim=True).view(features, -1)

        return dx, dweight.view_as(weight), dbias.view_as(bias), None

class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape).cuda())
        self.bias = nn.Parameter(torch.zeros(normalized_shape).cuda())
        self.eps = 1e-5

    def forward(self, x):
        return CustomLayerNorm.apply(x, self.weight, self.bias, self.eps)

def get_inputs():
    batch_size = 16
    features, dim1, dim2 = 64, 256, 256
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [(64, 256, 256)]