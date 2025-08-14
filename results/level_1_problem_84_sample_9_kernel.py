import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for depthwise 2D convolution
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int height_in,
    const int width_in,
    const int kernel_size,
    const int stride,
    const int pad_h,
    const int pad_w,
    const int height_out,
    const int width_out,
    const int out_channels) {

    int output_h = blockIdx y * blockDim.y + threadIdx.y;
    int output_w = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = threadIdx.z;

    if (output_h >= height_out || output_w >= width_out || channel >= out_channels) return;

    int input_h = output_h * stride - pad_h;
    int input_w = output_w * stride - pad_w;

    scalar_t sum = 0.0;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h = input_h + kh;
            int w = input_w + kw;
            if (h >= 0 && h < height_in && w >= 0 && w < width_in) {
                scalar_t val = input[channel * height_in * width_in + h * width_in + w];
                val *= weight[channel * kernel_size * kernel_size + kh * kernel_size + kw];
                sum += val;
            }
        }
    }

    output[channel * height_out * width_out + output_h * width_out + output_w] = sum;
}

at::Tensor depthwise_conv2d_cuda(
    at::Tensor input,
    at::Tensor weight,
    int stride,
    int padding) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height_in = input.size(2);
    const int width_in = input.size(3);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(0);

    int height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    int width_out = (width_in + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, out_channels, height_out, width_out}, input.options());

    dim3 threads(8, 8, 16); // Using 3D grid for channels
    dim3 blocks(
        (width_out + threads.x - 1) / threads.x,
        (height_out + threads.y - 1) / threads.y,
        (out_channels + threads.z - 1) / threads.z));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            height_in,
            width_in,
            kernel_size,
            stride,
            padding,
            padding, // Assuming symmetric padding
            height_out,
            width_out,
            out_channels);
    }));

    return output;
}
"""

depthwise_conv_cpp_source = """
at::Tensor depthwise_conv2d_cuda(
    at::Tensor input,
    at::Tensor weight,
    int stride,
    int padding);
"""

depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights similar to PyTorch's Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) 

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x):
        out = depthwise_conv.depthwise_conv2d_cuda(x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out