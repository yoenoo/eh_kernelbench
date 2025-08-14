import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import numpy as np

depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int output_height,
    int output_width) {

    const int HW = input_height * input_width;
    const int KW = kernel_w;
    const int KH = kernel_h;
    const int output_size = output_height * output_width;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ scalar_t smem[32 * 16][32 * 16]; // Adjust size according to possible max input tile

    for (int sample = 0; sample < batch_size; sample++) {
        for (int c = 0; c < in_channels; c++) {
            // Each thread processes one output element
            if (thread_id < output_size) {
                int output_x = thread_id % output_width;
                int output_y = thread_id / output_width;
                int in_x = output_x * stride_w - padding_w;
                int in_y = output_y * stride_h - padding_h;
                scalar_t sum = 0;
                for (int kh = 0; kh < kernel_h; ++kh) {
                    int y = in_y + kh * dilation_h;
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int x = in_x + kw * dilation_w;
                        // Check boundaries
                        if (y >= 0 && y < input_height && x >=0 && x < input_width) {
                            scalar_t w_val = weight[c * kernel_h * kernel_w + kh * kernel_w + kw];
                            scalar_t in_val = input[sample * in_channels * HW + c * HW + y * input_width + x];
                            sum += in_val * w_val;
                        }
                    }
                }
                output[sample * in_channels * output_height * output_width + c * output_height * output_width + output_y * output_width + output_x] = sum;
            }
        }
    }
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                                  int kernel_h, int kernel_w,
                                  int stride_h, int stride_w,
                                  int padding_h, int padding_w,
                                  int dilation_h, int dilation_w) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    // Compute output dimensions
    int output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch_size, in_channels, output_height, output_width}, input.options());

    int threads_per_block = 256;
    int blocks_per_grid = (output_height * output_width + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_cuda", ([&]{
        depthwise_conv2d_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
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

depthwise_conv_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int kernel_h, int kernel_w,
                                   int stride_h, int stride_w,
                                   int padding_h, int padding_w,
                                   int dilation_h, int dilation_w);
"""

depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_h, kernel_size_w,
                 stride_h=1, stride_w=1, padding_h=0, padding_w=0,
                 dilation_h=1, dilation_w=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        assert groups == in_channels, "Only supports depthwise (groups=in_channels)"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size_h, kernel_size_w)
        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)
        self.dilation = (dilation_h, dilation_w)
        self.groups = groups
        self.bias = bias
        # Initialize weights similar to PyTorch Conv2d
        self.weight = nn.Parameter(torch.empty((in_channels, kernel_size_h*kernel_size_w)))
        self.bias_term = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias_term is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias_term, -bound, bound)

    def forward(self, x):
        # Reshape weight to match expected by custom kernel (each channel's kernel flattened)
        return depthwise_conv.depthwise_conv2d_cuda(
            x, self.weight,
            self.kernel_size[0], self.kernel_size[1],
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )