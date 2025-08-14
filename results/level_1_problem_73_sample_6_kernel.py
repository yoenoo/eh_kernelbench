import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;     \
      i < (n);                                           \
      i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_size, int stride, int padding,
    int groups, int depth_out, int height_out, int width_out,
    int out_channels_per_group, int in_channels_per_group) {

  CUDA_1D_KERNEL_LOOP(output_index, batch_size * out_channels * depth_out * height_out * width_out) {
    int w_out = output_index % width_out;
    int h_out = (output_index / width_out) % height_out;
    int d_out = (output_index / (width_out * height_out)) % depth_out;
    int group_out = (output_index / (width_out * height_out * depth_out)) % groups;
    int batch = output_index / (width_out * height_out * depth_out * groups);

    int out_channel_idx = (output_index / (width_out * height_out * depth_out * groups)) * out_channels_per_group
                         + (output_index % (out_channels_per_group * width_out * height_out * depth_out))
                           / (width_out * height_out * depth_out);

    scalar_t val = 0;

    for (int kd = 0; kd < kernel_size; ++kd) {
      for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
          // Compute input indices
          int d_in = (d_out * stride - padding) - kd;
          int h_in = (h_out * stride - padding) - kh;
          int w_in = (w_out * stride - padding) - kw;

          // Check if within input bounds
          if (d_in < 0 || d_in >= depth || h_in < 0 || h_in >= height || w_in < 0 || w_in >= width)
            continue;

          for (int c_in_group = 0; c_in_group < in_channels_per_group; ++c_in_group) {
            int in_channel = group_out * in_channels_per_group + c_in_group;
            int weight_offset = (group_out * out_channels_per_group + out_channel_idx) * kernel_size*kernel_size*kernel_size*in_channels_per_group
                               + c_in_group * kernel_size*kernel_size*kernel_size
                               + kd * kernel_size*kernel_size + kh * kernel_size + kw;

            int input_offset = batch * in_channels * depth * height * width
                              + in_channel * depth * height * width
                              + d_in * height * width + h_in * width + w_in;

            val += input[input_offset] * weight[weight_offset];
          }
        }
      }
    }
    // Compute output index correctly
    int out_channel = group_out * out_channels_per_group + out_channel_idx;
    int output_offset = batch * out_channels * depth_out * height_out * width_out
                       + out_channel * depth_out * height_out * width_out
                       + d_out * height_out * width_out + h_out * width_out + w_out;
    output[output_offset] = val;
  }
}

at::Tensor conv_transpose3d_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    int stride, int padding, int groups,
    int kernel_size,
    int depth_out, int height_out, int width_out) {

  const auto batch_size = input.size(0);
  const auto in_channels = input.size(1);
  const auto out_channels = weight.size(0); // assuming weight is [out_channels, ...]
  const auto depth = input.size(2);
  const auto height = input.size(3);
  const auto width = input.size(4);

  auto output = at::empty({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

  int out_channels_per_group = out_channels / groups;
  int in_channels_per_group = in_channels / groups;

  dim3 blocks(ATen::cuda::getHipBlockCount(output.numel(), 256));
  dim3 threads(256);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_cuda", ([&] {
    conv_transpose3d_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data<scalar_t>(),
        weight.data<scalar_t>(),
        output.data<scalar_t>(),
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size, stride, padding,
        groups,
        depth_out, height_out, width_out,
        out_channels_per_group, in_channels_per_group);
  }));

  cudaDeviceSynchronize();
  return output;
}
"""

conv_transpose3d_cpp_source = """
at::Tensor conv_transpose3d_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    int stride, int padding, int groups,
    int kernel_size,
    int depth_out, int height_out, int width_out);
"""

# Compile the inline CUDA code
conv_transpose3d = load_inline(
    name='conv_transpose3d',
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=['conv_transpose3d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias

        # Initialize weight (assuming similar to ConvTranspose3d)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output dimensions (simplified based on input assumptions)
        depth = x.size(2)
        height = x.size(3)
        width = x.size(4)
        depth_out = (depth - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        height_out = (height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        width_out = (width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        return conv_transpose3d.conv_transpose3d_cuda(
            x, self.weight, self.stride, self.padding, self.groups,
            self.kernel_size, depth_out, height_out, width_out)