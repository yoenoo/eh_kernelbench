import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

class _ConvTranspose2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, output_padding, groups, output_padding_h):
        # Custom forward pass using CUDA kernel
        batch_size, in_channels, input_h, input_w = input.shape
        kernel_h, kernel_w = weight.shape[2], weight.shape[3]
        out_channels = weight.shape[0]
        
        # Calculating output dimensions
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation
        output_padding_h, output_padding_w = output_padding

        output_h = (input_h - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1
        output_w = (input_w - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1
        
        output = torch.zeros(batch_size, out_channels, output_h, output_w, device=input.device, dtype=input.dtype)

        # Launching CUDA kernel
        threads_per_block = (16, 16)
        blocks_per_grid = (
            math.ceil(output_w / threads_per_block[1]),
            math.ceil(output_h / threads_per_block[0]),
            batch_size * out_channels
        )

        # Save context for backward
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.output_padding = output_padding
        ctx.groups = groups
        
        # Kernel call here (simplified, to be implemented)
        # conv_transpose2d_kernel <<<blocks_per_grid, threads_per_block>>> (
        #     input, weight, bias, output, stride, padding, dilation, output_padding
        # )
        # Note: Actual kernel implementation requires handling spatial dimensions and tensor storage details.
        # This is a placeholder for the real kernel.
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Implement backward pass here using custom CUDA kernels
        input, weight, bias = ctx.saved_tensors
        # ...
        # The backward pass should also be implemented with custom kernels
        return grad_output, None, None, None, None, None, None, None, None

class ConvTranspose2dCustom(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(ConvTranspose2dCustom, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = 1
        self.output_padding = (0, 0)
        
        # Weight and bias initialization
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // self.groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return _ConvTranspose2dFunction.apply(
            x, 
            self.weight, 
            self.bias if self.bias is not None else torch.zeros(1),
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
            self.groups,
            self.output_padding[0]
        )

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = ConvTranspose2dCustom(
            in_channels, out_channels, kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            bias=bias
        )

    def forward(self, x):
        return self.conv_transpose2d(x)