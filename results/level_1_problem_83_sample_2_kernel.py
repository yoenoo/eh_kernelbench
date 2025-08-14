import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Each thread computes one output element
    const int ow = blockIdx.x * blockDim.x + tx;
    const int oh = blockIdx.y * blockDim.y + ty;
    const int channel = blockIdx.z;

    if (ow >= output_width || oh >= output_height) {
        return;
    }

    scalar_t sum = 0;
    for (int kw = 0; kw < kernel_size; ++kw) {
        const int iw = ow * stride - padding + kw * dilation;
        if (iw < 0 || iw >= input_width) {
            continue;
        }

        for (int kh = 0; kh < kernel_size; ++kh) {
            const int ih = oh * stride - padding + kh * dilation;
            if (ih < 0 || ih >= input_height) {
                continue;
            }
            
            const int input_idx = channel * input_height * input_width + ih * input_width + iw;
            const int weight_idx = kh * kernel_size + kw;
            
            sum += input[input_idx] * weight[weight_idx];
        }
    }

    const int output_idx = channel * output_height * output_width + oh * output_width + ow;
    output[output_idx] = sum;
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding, int dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    torch::Tensor output = torch::empty({batch_size, in_channels, output_height, output_width}, input.options());

    dim3 threads(32, 8);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        in_channels);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            input_height,
            input_width,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    return output;
}
"""

depthwise_conv_cpp_source = "torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding, int dilation);"

depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=[depthwise_conv_cpp_source],
    cuda_sources=[depthwise_conv_source],
    functions=["depthwise_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty((in_channels, kernel_size * kernel_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None
        self.cuda_conv = depthwise_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cuda_conv.depthwise_conv2d_cuda(
            x,
            self.weight.view(self.in_channels, self.kernel_size, self.kernel_size),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out