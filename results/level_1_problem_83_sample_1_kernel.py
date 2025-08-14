import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                           \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

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

  CUDA_1D_KERNEL_LOOP(output_index, batch_size * in_channels * input_height * input_width) {
    int width = input_width;
    int width_pad = width + 2 * padding;
    int kernel_radius = (kernel_size - 1) / 2;

    int w_stride = 1;
    int h_stride = stride;

    int w_dilation = 1;
    int h_dilation = dilation;

    int channel = output_index % in_channels;
    int batch = output_index / (in_channels * input_height * input_width);
    int output_pos = output_index / in_channels;
    int output_h = output_pos / input_width;
    int output_w = output_pos % input_width;

    int input_h_start = output_h * h_stride - padding;
    int input_w_start = output_w * w_stride - padding;

    scalar_t val = 0;
    for (int k = 0; k < kernel_size; ++k) {
      int input_h = input_h_start + k * h_dilation;
      if (input_h < 0 || input_h >= input_height) continue;
      int input_offset = batch * in_channels * input_height * input_width +
                        channel * input_height * input_width +
                        input_h * input_width + output_w;
      int weight_offset = channel * kernel_size + k;
      val += input[input_offset] * weight[weight_offset];
    }
    output[output_index] = val;
  }
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, 
                                   int stride, int padding, int dilation) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);
    const auto kernel_size = weight.size(1); // Assuming kernel is (K,1)

    auto output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1;
    auto output_width = input_width;

    auto output = torch::zeros({batch_size, in_channels, output_height, output_width}, input.options());

    dim3 blocks(16, 16);
    dim3 threads(32, 8);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, input_height, input_width,
            kernel_size, stride, padding, dilation);
    }));

    return output;
}
"""

cpp_source = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, 
                                   int stride, int padding, int dilation);
"""

depthwise_conv_op = load_inline(
    name="depthwise_conv",
    cpp_sources=cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # Initialize weights similar to PyTorch's Conv2d
        self.weight = nn.Parameter(torch.empty(in_channels, kernel_size, 1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
        self.depthwise_conv = depthwise_conv_op

    def forward(self, x):
        output = self.depthwise_conv.depthwise_conv2d_cuda(
            x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output