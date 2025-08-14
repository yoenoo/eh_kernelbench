import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ weight,
                                       scalar_t* __restrict__ output,
                                       const int batch_size,
                                       const int in_channels,
                                       const int input_height,
                                       const int input_width,
                                       const int kernel_h,
                                       const int kernel_w,
                                       const int stride_h,
                                       const int stride_w,
                                       const int padding_h,
                                       const int padding_w,
                                       const int dilation_h,
                                       const int dilation_w,
                                       const int output_height,
                                       const int output_width) {
    const int output_size = batch_size * in_channels * output_height * output_width;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_size) return;

    const int w = index % output_width;
    const int h = (index / output_width) % output_height;
    const int c = (index / output_width / output_height) % in_channels;
    const int n = index / output_width / output_height / in_channels;

    scalar_t val = 0;
    for (int ky = 0; ky < kernel_h; ++ky) {
        for (int kx = 0; kx < kernel_w; ++kx) {
            const int y_off = h * stride_h + ky * dilation_h - padding_h;
            const int x_off = w * stride_w + kx * dilation_w - padding_w;
            if (y_off >= 0 && y_off < input_height && x_off >= 0 && x_off < input_width) {
                const int input_idx = n * in_channels * input_height * input_width + 
                                      c * input_height * input_width + 
                                      y_off * input_width + x_off;
                const int weight_idx = c * kernel_h * kernel_w + ky * kernel_w + kx;
                val += input[input_idx] * weight[weight_idx];
            }
        }
    }
    output[index] = val;
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int stride_h,
                                   int stride_w,
                                   int padding_h,
                                   int padding_w,
                                   int dilation_h,
                                   int dilation_w) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int kernel_h = weight.size(1);
    const int kernel_w = weight.size(2);

    // Compute output dimensions
    int output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, in_channels, output_height, output_width}, 
                              torch::device(input.device()).dtype(input.dtype()));

    const int threads = 256;
    const int output_size = batch_size * in_channels * output_height * output_width;
    const int blocks = (output_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            input_height,
            input_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            output_height,
            output_width);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

depthwise_conv_cpp_source = (
    "torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);"
)

depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=[depthwise_conv_cpp_source],
    cuda_sources=[depthwise_conv_source],
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lm"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Original parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.bias = bias
        
        # Initialize weight
        self.weight = nn.Parameter(torch.empty(in_channels, kernel_size_h, kernel_size_w))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return depthwise_conv.depthwise_conv2d_cuda(x, self.weight, self.stride_h, self.stride_w, self.padding_h, self.padding_w, self.dilation_h, self.dilation_w)