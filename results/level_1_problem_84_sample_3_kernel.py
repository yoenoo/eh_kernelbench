import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for depthwise convolution
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \\
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \\
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void depthwise_conv2d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int batch_size,
    int in_channels,
    int out_channels_per_group,
    int height_in,
    int width_in,
    int kernel_size,
    int height_out,
    int width_out,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w) {
  
  CUDA_1D_KERNEL_LOOP(index, batch_size * in_channels * height_out * width_out) {
    int w_out = index % width_out;
    int h_out = (index / width_out) % height_out;
    int channel = (index / (height_out * width_out)) % in_channels;
    int n = index / (in_channels * height_out * width_out);

    scalar_t sum = 0;
    for (int kh = 0; kh < kernel_size; ++kh) {
      for (int kw = 0; kw < kernel_size; ++kw) {
        int h_in = h_out * stride_h - pad_h + kh;
        int w_in = w_out * stride_w - pad_w + kw;
        if (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
          sum += input[n][channel][h_in][w_in] * 
                 weight[channel][0][kh][kw];
        }
      }
    }
    output[n][channel][h_out][w_out] = sum;
  }
}

torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w) {
  
  // Compute output dimensions
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int height_in = input.size(2);
  int width_in = input.size(3);
  int kernel_size = weight.size(2); // Assuming square kernel

  int height_out = (height_in + 2 * pad_h - kernel_size) / stride_h + 1;
  int width_out = (width_in + 2 * pad_w - kernel_size) / stride_w + 1;
  
  auto output = torch::zeros({batch_size, in_channels, height_out, width_out}, 
                            input.options());

  dim3 blocks_per_grid((batch_size * in_channels * height_out * width_out + 256 - 1) / 256);
  dim3 threads_per_block(256);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_forward", ([&] {
    depthwise_conv2d_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
      input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
      weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
      output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
      batch_size,
      in_channels,
      weight.size(1)/in_channels, // out_channels_per_group (1 for depthwise)
      height_in,
      width_in,
      kernel_size,
      height_out,
      width_out,
      stride_h,
      stride_w,
      pad_h,
      pad_w);
  }));
  
  cudaDeviceSynchronize();
  return output;
}
"""

# Compile the CUDA kernel
depthwise_conv = load_inline(
    name='depthwise_conv',
    cuda_sources=depthwise_conv_source,
    functions=['depthwise_conv2d_forward'],
    verbose=True,
    extra_cflags=['-std=c++14'],
    extra_cuda_cflags=['-std=c++14', '--expt-relaxed-constexpr']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # No bias support as per original implementation
        # if bias:
        #     self.bias = nn.Parameter(torch.empty(out_channels))
        # else:
        #     self.register_parameter('bias', None)
        # self.bias = None
        
        self.depthwise_conv = depthwise_conv
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise_conv.depthwise_conv2d_forward(
            x,
            self.weight,
            self.stride,
            self.stride,
            self.padding,
            self.padding
        )