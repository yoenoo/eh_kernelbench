cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for depthwise separable convolution
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ weight,
                                       scalar_t* output,
                                       const int batches,
                                       const int in_channels,
                                       const int input_height,
                                       const int input_width,
                                       const int kernel_size,
                                       const int stride,
                                       const int padding,
                                       const int dilation) {

    // Your depthwise convolution kernel implementation here
    // Implement efficient tiling and shared memory for better performance

    // Indices and offsets
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    const int b = blockIdx.z;

    if (n >= output_height * output_width || c >= in_channels || b >= batches) return;

    // Compute output spatial indices
    int ow = n % output_width;
    int oh = n / output_width;

    // Compute input spatial indices
    int hstart = -padding + oh * stride;
    int wstart = -padding + ow * stride;

    scalar_t val = 0;
    for (int kh = 0; kh < kernel_size; ++kh) {
        const int h = hstart + dilation * kh;
        if (h < 0 || h >= input_height) continue;

        for (int kw = 0; kw < kernel_size; ++kw) {
            const int w = wstart + dilation * kw;
            if (w < 0 || w >= input_width) continue;

            val += input[b * in_channels * input_height * input_width + 
                        c * input_height * input_width + 
                        h * input_width + w] *
                   weight[c * kernel_size * kernel_size + kh * kernel_size + kw];
        }
    }

    output[b * in_channels * output_height * output_width + 
           c * output_height * output_width + 
           oh * output_width + ow] = val;
}

torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
) {
    const auto batches = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);
    const auto kernel_size = weight.size(2);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::zeros({batches, in_channels, output_height, output_width}, input.options());

    dim3 threads(256);
    dim3 blocks(output_height * output_width, in_channels, batches);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_forward", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batches, in_channels, input_height, input_width,
            kernel_size, stride, padding, dilation);
    }));

    return output;
}
"""

pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void pointwise_conv2d_kernel(const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ weight,
                                       scalar_t* output,
                                       const int batches,
                                       const int in_channels,
                                       const int out_channels,
                                       const int height,
                                       const int width) {

    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int w = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z / out_channels;
    const int oc = blockIdx.z % out_channels;

    if (h >= height || w >= width || b >= batches || oc >= out_channels) return;

    scalar_t val = 0;
    for (int ic = 0; ic < in_channels; ++ic) {
        val += input[b * in_channels * height * width + ic * height * width + h * width + w] *
               weight[oc * in_channels + ic];
    }
    output[b * out_channels * height * width + oc * height * width + h * width + w] = val;
}

torch::Tensor pointwise_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight
) {
    const auto batches = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0);
    const auto height = input.size(2);
    const auto width = input.size(3);

    auto output = torch::zeros({batches, out_channels, height, width}, input.options());

    dim3 threads(16, 16, 1);
    dim3 blocks((height + 15) / 16, (width + 15) / 16, (batches * out_channels + 255) / 256);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "pointwise_conv2d_forward", ([&] {
        pointwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batches, in_channels, out_channels, height, width);
    }));

    return output;
}
"""

# Compile depthwise and pointwise convolution kernels
depthwise_conv_module = load_inline(
    name='depthwise_conv',
    cuda_sources=depthwise_conv_source,
    functions=['depthwise_conv2d_forward'],
    verbose=True
)
pointwise_conv_module = load_inline(
    name='pointwise_conv',
    cuda_sources=pointwise_conv_source,
    functions=['pointwise_conv2d_forward'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.depthwise_weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        self.pointwise_weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize weights using the same method as PyTorch
        nn.init.kaiming_uniform_(self.depthwise_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise_weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.pointwise_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        depthwise_out = depthwise_conv_module.depthwise_conv2d_forward(
            x, self.depthwise_weight, self.stride, self.padding, self.dilation)
        pointwise_out = pointwise_conv_module.pointwise_conv2d_forward(
            depthwise_out.view(depthwise_out.size(0), depthwise_out.size(1), 
                              depthwise_out.size(2), depthwise_out.size(3)),
            self.pointwise_weight.view(self.pointwise_weight.size(0), self.pointwise_weight.size(1)))
        if self.bias is not None:
            pointwise_out += self.bias.view(1, -1, 1, 1)
        return pointwise_out