import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class DepthwiseConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation):
        # Save context for backward pass
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.save_for_backward(input, weight, bias)
        
        # Call CUDA kernel for forward pass
        output = depthwise_conv2d_forward(input, weight, stride, padding, dilation)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # Compute gradient w.r.t input
            grad_input = depthwise_conv2d_backward_input(
                input.shape, weight, grad_output, ctx.stride, ctx.padding, ctx.dilation
            )
        if ctx.needs_input_grad[1]:
            # Compute gradient w.r.t weight
            grad_weight = depthwise_conv2d_backward_weight(
                weight.shape, input, grad_output, ctx.stride, ctx.padding, ctx.dilation
            )
        if bias is not None and ctx.needs_input_grad[2]:
            # Compute gradient w.r.t bias
            grad_bias = grad_output.sum((0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None

# Define CUDA kernels for forward, backward_input, backward_weight
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void depthwise_conv2d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    const int kernel_size, const int stride, const int padding, const int dilation)
{
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int output_height = output.size(2);
    const int output_width = output.size(3);

    CUDA_1D_KERNEL_LOOP(output_idx, batch_size * channels * output_height * output_width) {
        int w = output_idx % output_width;
        int h = (output_idx / output_width) % output_height;
        int c = (output_idx / (output_width * output_height)) % channels;
        int n = output_idx / (channels * output_width * output_height);

        scalar_t val = 0;
        for (int kh = 0; kh < kernel_size; ++kh) {
            int ih = h * stride + kh * dilation - padding;
            for (int kw = 0; kw < 1; ++kw) { // Only kernel size in height direction
                int iw = w * stride + kw * dilation - padding;
                // Check if indices are within input boundaries
                if (ih >=0 && ih < input_height && iw >=0 && iw < input_width) {
                    val += input[n][c][ih][iw] * weight[c][kh][kw];
                }
            }
        }
        output[n][c][h][w] = val;
    }
}

torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation)
{
    // Calculate output dimensions
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    auto kernel_size = weight.size(1);
    
    // Calculate output size
    auto output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto output_width = (input_width + 2 * padding - dilation * (1 - 1) - 1) / stride + 1; // Since kernel_width is 1

    torch::Tensor output = torch::zeros({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int elements = batch_size * channels * output_height * output_width;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_forward", ([&] {
        depthwise_conv2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,3,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            kernel_size,
            stride,
            padding,
            dilation
        );
    }));

    cudaDeviceSynchronize();
    return output;
}

// Backward kernels (simplified for brevity, implementation would depend on actual requirements and optimization strategy)
// ... similar kernels for backward operations
"""

# Compile the CUDA kernel
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp Sources="",
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_forward"],
    verbose=True
)

depthwise_conv2d_forward = depthwise_conv2d.depthwise_conv2d_forward

class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # Initialize weights similar to PyTorch's Conv2d
        self.weight = nn.Parameter(torch.Tensor(in_channels, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return DepthwiseConv2dFunction.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation)