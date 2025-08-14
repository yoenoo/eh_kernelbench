import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Convolution.h>

using torch::Tensor;

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);    \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void custom_conv3d_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int64_t batches,
    int64_t input_channels,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t kernel_depth,
    int64_t stride_h,
    int64_t stride_w,
    int64_t stride_d,
    int64_t padding_h,
    int64_t padding_w,
    int64_t padding_d,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t dilation_d,
    int64_t output_height,
    int64_t output_width,
    int64_t output_depth) {

  CUDA_1D_KERNEL_LOOP(output_index, batches * output_channels * output_depth * output_height * output_width) {
    int batch_idx = output_index / (output_channels * output_depth * output_height * output_width);
    int channel_out = (output_index / (output_depth * output_height * output_width)) % output_channels;
    int d_out = (output_index / (output_height * output_width)) % output_depth;
    int h_out = (output_index / output_width) % output_height;
    int w_out = output_index % output_width;

    int channel_in = channel_out; // assuming groups = 1

    scalar_t sum = 0;
    for (int k_depth = 0; k_depth < kernel_depth; ++k_depth) {
        int d_in = d_out * stride_d - padding_d + k_depth * dilation_d;
        if (d_in < 0 || d_in >= input_depth) continue;
        for (int k_h = 0; k_h < kernel_height; ++k_h) {
            int h_in = h_out * stride_h - padding_h + k_h * dilation_h;
            if (h_in < 0 || h_in >= input_height) continue;
            for (int k_w = 0; k_w < kernel_width; ++k_w) {
                int w_in = w_out * stride_w - padding_w + k_w * dilation_w;
                if (w_in < 0 || w_in >= input_width) continue;
                
                scalar_t w_val = weight[channel_out * (kernel_height * kernel_width * kernel_depth) + (k_h * kernel_width + k_w) * kernel_depth + k_depth];
                scalar_t in_val = input[batch_idx * input_channels * input_depth * input_height * input_width +
                                      channel_in * input_depth * input_height * input_width +
                                      d_in * input_height * input_width +
                                      h_in * input_width + w_in];
                sum += in_val * w_val;
            }
        }
    }
    output[output_index] = sum;
  }
}

at::Tensor custom_conv3d_forward(
    at::Tensor input,
    at::Tensor weight,
    int stride_h,
    int stride_w,
    int stride_d,
    int padding_h,
    int padding_w,
    int padding_d,
    int dilation_h,
    int dilation_w,
    int dilation_d) {

  const auto batches = input.size(0);
  const auto input_channels = input.size(1);
  const auto input_depth = input.size(2);
  const auto input_height = input.size(3);
  const auto input_width = input.size(4);

  const auto output_channels = weight.size(0); // since weight is [out_channels, in_channels/groups, ...]
  const auto kernel_height = weight.size(2);
  const auto kernel_width = weight.size(3);
  const auto kernel_depth = weight.size(4);

  // Compute output dimensions
  const int output_height = (input_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
  const int output_width = (input_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;
  const int output_depth = (input_depth + 2 * padding_d - dilation_d * (kernel_depth - 1) - 1) / stride_d + 1;

  at::Tensor output = at::empty({batches, output_channels, output_depth, output_height, output_width}, input.options());

  dim3 blocks(DIVUP(batches * output_channels * output_depth * output_height * output_width, 512));
  dim3 threads(512);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv3d_forward", ([&] {
      custom_conv3d_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          input.data<scalar_t>(),
          weight.data<scalar_t>(),
          output.data<scalar_t>(),
          batches,
          input_channels,
          input_depth,
          input_height,
          input_width,
          output_channels,
          kernel_height,
          kernel_width,
          kernel_depth,
          stride_h,
          stride_w,
          stride_d,
          padding_h,
          padding_w,
          padding_d,
          dilation_h,
          dilation_w,
          dilation_d,
          output_height,
          output_width,
          output_depth);
  }));

  return output;
}
"""

conv3d_cpp_source = """
at::Tensor custom_conv3d_forward(
    at::Tensor input,
    at::Tensor weight,
    int stride_h,
    int stride_w,
    int stride_d,
    int padding_h,
    int padding_w,
    int padding_d,
    int dilation_h,
    int dilation_w,
    int dilation_d);
"""

custom_conv3d = load_inline(
    name="custom_conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_kernel_source,
    functions=["custom_conv3d_forward"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = (stride, stride, 1) if isinstance(stride, int) else stride
        self.padding = (padding, padding, 0) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation, 1) if isinstance(dilation, int) else dilation
        self.kernel_size = (kernel_size, kernel_size, 1)
        self.groups = groups

        # Initialize weights similar to PyTorch's Conv3d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = custom_conv3d.custom_conv3d_forward(
            x,
            self.weight,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.dilation[0],
            self.dilation[1],
            self.dilation[2]
        )
        return output