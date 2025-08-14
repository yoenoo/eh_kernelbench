import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom Conv3d implementation with optimized CUDA kernel
conv3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define CUDA_3D_KERNEL_LOOP(i, n)                       \
  for (int i = blockIdx.z * blockDim.z + threadIdx.z; i < n; i += blockDim.z * gridDim.z)

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename T>
__device__ inline TatomicAdd(T* address, T val) {
  return atomicAdd(address, val);
}

template <typename T>
__global__ void conv3d_forward_kernel(const T* __restrict__ input, const T* __restrict__ weight, T* __restrict__ output,
    int b, int in_c, int out_c, int input_d, int input_h, int input_w,
    int kernel_h, int kernel_w, int kernel_t,
    int padding_h, int padding_w, int padding_t,
    int stride_h, int stride_w, int stride_t,
    int dilation_h, int dilation_w, int dilation_t) {

  CUDA_3D_KERNEL_LOOP(index, b * out_c * output_d * output_h * output_w) {
    int w_out = index % output_w;
    int h_out = (index / output_w) % output_h;
    int d_out = (index / (output_w * output_h)) % output_d;
    int c_out = (index / (output_w * output_h * output_d)) % out_c;
    int n     = index / (out_c * output_d * output_h * output_w);

    T val = 0;
    int d_in_start = d_out * stride_t - padding_t;
    int h_in_start = h_out * stride_h - padding_h;
    int w_in_start = w_out * stride_w - padding_w;

    for (int i = 0; i < kernel_t; ++i) {
      int d_in = d_in_start + i * dilation_t;
      if (d_in < 0 || d_in >= input_d) continue;
      for (int j = 0; j < kernel_h; ++j) {
        int h_in = h_in_start + j * dilation_h;
        if (h_in < 0 || h_in >= input_h) continue;
        for (int k = 0; k < kernel_w; ++k) {
          int w_in = w_in_start + k * dilation_w;
          if (w_in < 0 || w_in >= input_w) continue;
          for (int c_in = 0; c_in < in_c; ++c_in) {
            const T w_val = weight[c_out * in_c * kernel_h * kernel_w * kernel_t + c_in * kernel_h * kernel_w * kernel_t + j * kernel_w * kernel_t + k * kernel_t + i];
            const T in_val = input[n * in_c * input_h * input_w * input_d + c_in * input_h * input_w * input_d + h_in * input_w * input_d + w_in * input_d + d_in];
            val += in_val * w_val;
          }
        }
      }
    }
    output[index] = val;
  }
}

torch::Tensor conv3d_forward(torch::Tensor input, torch::Tensor weight,
    int kernel_size, int padding_t, int stride_t,
    int dilation_t) {

  // Deduce dimensions
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int input_depth = input.size(2);
  int input_height = input.size(3);
  int input_width = input.size(4);
  int out_channels = weight.size(0);
  int kernel_height = kernel_size;
  int kernel_width = kernel_size;
  int kernel_depth = 1; // Since kernel_size is for 2D, third dim is 1

  // Output dimensions calculation
  int output_depth = (input_depth + 2 * padding_t - dilation_t * (kernel_depth - 1) - 1) / stride_t + 1;
  int output_height = (input_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
  int output_width = (input_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;

  // Create output tensor
  auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

  // Launch kernel
  dim3 threads(256);
  dim3 blocks((output_depth * output_height * output_width * out_channels + threads.x - 1) / threads.x);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_forward", ([&] {
    conv3d_forward_kernel<scalar_t><<<blocks, threads>>>(
      input.data<scalar_t>(), weight.data<scalar_t>(), output.data<scalar_t>(),
      batch_size, in_channels, out_channels,
      input_depth, input_height, input_width,
      kernel_height, kernel_width, kernel_depth,
      padding_h, padding_w, padding_t,
      stride_h, stride_w, stride_t,
      dilation_h, dilation_w, dilation_t);
  }));

  return output;
}
"""

conv3d_cpp_source = (
    "torch::Tensor conv3d_forward(torch::Tensor input, torch::Tensor weight, "
    "int kernel_size, int padding_t, int stride_t, int dilation_t);"
)

# Compile the custom 3D convolution CUDA kernel
conv3d_cuda = load_inline(
    name="conv3d_cuda",
    cpp_sources=[conv3d_cpp_source],
    cuda_sources=[conv3d_cuda_source],
    functions=["conv3d_forward"],
    verbose=True,
    extra_cuda_cflags=["-arch=sm_70"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride, 1) if isinstance(stride, int) else stride
        self.padding = (padding, padding, 0) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation, 1) if isinstance(dilation, int) else dilation
        self.groups = groups
        
        # Initialize weights (assuming square kernel in spatial dims)
        weight_size = (out_channels, in_channels // groups, kernel_size, kernel_size, 1)
        self.weight = nn.Parameter(torch.empty(weight_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Prepare parameters
        stride_t = self.stride[2] if len(self.stride) ==3 else 1
        padding_t = self.padding[2] if len(self.padding) ==3 else 0
        dilation_t = self.dilation[2] if len(self.dilation) ==3 else 1
        
        output = conv3d_cuda.conv3d_forward(
            x.contiguous(),
            self.weight.contiguous(),
            self.kernel_size,
            padding_t,
            stride_t,
            dilation_t
        )
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        
        return output