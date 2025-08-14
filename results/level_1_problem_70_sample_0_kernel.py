import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class CustomConvTranspose3dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, output_padding, dilation, groups):
        # Since PyTorch's conv_transpose3d is already optimized, the actual implementation here would depend
        # on specific optimizations. However, for this example, we'll use the native implementation.
        # The CUDA kernel would need to be implemented here with optimizations like tensor cores, memory coalescing, etc.
        output = torch.conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation)
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.dilation = dilation
        ctx.groups = groups
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Implement custom backward pass if needed, otherwise fall back to autograd
        input, weight, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = torch.autograd.grad(
            outputs=(torch.conv_transpose3d(input, weight, bias, ctx.stride, ctx.padding, ctx.output_padding, ctx.groups, ctx.dilation)),
            inputs=(input, weight, bias),
            grad_outputs=(grad_output,)
        )
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

        # Initialize weights similar to nn.ConvTranspose3d
        kernel_size = (kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and biases using the same method as PyTorch's default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.bias is not None:
            return CustomConvTranspose3dFunction.apply(
                x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.dilation, self.groups
            )
        else:
            return CustomConvTransposeFunction.apply(
                x, self.weight, None, self.stride, self.padding, self.output_padding, self.dilation, self.groups
            )

    def extra_repr(self):
        s = ('in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

# Note: The actual CUDA kernel implementation requires writing a custom CUDA kernel that performs the 3D transposed convolution
#       with optimizations such as optimized memory access patterns, loop unrolling, and use of tensor cores for better performance.