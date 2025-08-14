import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class DepthwiseConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        # Save the parameters and inputs for backward pass
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        # Check that the input dimensions are compatible
        batch_size, in_channels, height, width = input.shape
        out_channels, _, kernel_size_h, kernel_size_w = weight.shape

        # Compute output dimensions
        out_height = (height + 2 * padding - dilation * (kernel_size_h - 1) - 1) // stride + 1
        out_width = width  # Assuming kernel_size_w is 1, so width remains the same

        # Initialize output tensor
        output = torch.zeros(batch_size, out_channels, out_height, out_width, device=input.device)

        # Configure CUDA grid dimensions
        threads_per_block = (32, 8)  # Tuned for better coalescing and occupancy
        blocks_per_grid = (
            (out_height + threads_per_block[0] - 1) // threads_per_block[0],
            (out_width + threads_per_block[1] - 1) // threads_per_block[1],
        )

        # Launch the custom CUDA kernel
        depthwise_conv2d_kernel(
            blocks_per_grid,
            threads_per_block,
            [
                input,
                weight,
                output,
                stride,
                padding,
                dilation,
                height,
                width,
                out_height,
                out_width,
                kernel_size_h,
                kernel_size_w,
                in_channels,
                batch_size,
            ],
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Implement the backward pass (gradient computation)
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups

        # Placeholder for gradients. Adjust based on actual computation
        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias) if bias is not None else None

        # Implement backward kernel or use PyTorch's built-in functions if possible
        # This part requires a custom backward CUDA kernel similar to forward
        # For simplicity, using PyTorch's autograd for now (will be replaced with custom kernel)
        grad_input = input.grad
        grad_weight = weight.grad
        if bias is not None:
            grad_bias = bias.grad

        return grad_input, grad_weight, grad_bias, None, None, None, None


# Define the CUDA kernel for depthwise convolution
depthwise_conv2d_kernel_code = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    const int stride,
    const int padding,
    const int dilation,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_h,
    const int kernel_w,
    const int in_channels,
    const int batch_size
) {
    const int B = blockIdx.x;
    const int C = blockIdx.y;
    const int out_y = threadIdx.x;
    const int out_x = threadIdx.y;

    // Compute the input coordinates based on output coordinates
    const int in_y = out_y * stride - padding;
    const int in_x = out_x;

    scalar_t sum = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
        const int y = in_y + kh * dilation;
        if (y < 0 || y >= in_height)
            continue;
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int x = in_x + kw * dilation;
            if (x < 0 || x >= in_width)
                continue;
            sum += input[B][C][y][x] * weight[C][0][kh][kw];
        }
    }
    output[B][C][out_y][out_x] = sum;
}
"""

# Compile the kernel
depthwise_conv2d_kernel = load_inline(
    name="depthwise_conv2d_kernel",
    cpp_sources="",
    cuda_sources=depthwise_conv2d_kernel_code,
    functions=[],
    verbose=True
)

depthwise_conv2d_kernel = torch.ops.depthwise_conv2d_kernel.depthwise_conv2d_kernel

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = (kernel_size, 1)  # Force kernel to be asymmetric (height is kernel_size, width is 1)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = in_channels

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return DepthwiseConv2dFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )