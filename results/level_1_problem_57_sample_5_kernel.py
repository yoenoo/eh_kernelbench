import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/cuda/Convolution.cuh>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
void conv_transpose2d_kernelLauncher(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int64_t in_channels, int64_t out_channels,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t padding_h, int64_t padding_w,
    int64_t output_padding_h, int64_t output_padding_w,
    int64_t groups) {

  // Implementing efficient conv transpose requires complex CUDA parallelism and memory management.
  // This requires handling the mathematics of transposed convolution through CUDA threads and blocks.
  // The kernel will need to loop over output elements, compute the corresponding input regions, and accumulate weights.

  // This is a simplified version to demonstrate structure; in practice, this requires a complete implementation
  // including handling dilation, group convolution, bias addition, and optimized memory access patterns.

  const int batch = input.size(0);
  const int input_height = input.size(2);
  const int input_width = input.size(3);

  const int output_height = output.size(2);
  const int output_width = output.size(3);

  const int channels_per_group = in_channels / groups;

  // Grid and block dimensions
  int block_size = 1024;
  dim3 block(block_size);
  dim3 grid(batch * out_channels * output_height * output_width);

  AT_CUDA_CHECK(cudaGetLastError());

  // Perform the actual computation here. This would involve:
  // For each output element, determine the input region that contributes to it via transpose logic
  // Iterate over the kernel elements and accumulate the contributions with appropriate indexing

  // Example pseudocode for kernel:
  auto conv_transpose_op = [=] __device__ (int idx) {
      int output_w = idx % output_width;
      int output_h = (idx / output_width) % output_height;
      int oc = (idx / (output_width * output_height)) % out_channels;
      int n = idx / (out_channels * output_width * output_height);

      int ic_group = oc / channels_per_group;
      int ic = oc % channels_per_group;

      int weight_offset = (ic_group * in_channels + ic) * kernel_h * kernel_w;

      for (int kh = 0; kh < kernel_h; ++kh) {
          for (int kw = 0; kw < kernel_w; ++kw) {
              // Compute input coordinates based on transposed convolution
              int input_h = (output_h - output_padding_h - kh) / stride_h + padding_h;
              int input_w = (output_w - output_padding_w - kw) / stride_w + padding_w;

              if ((input_h >=0) && (input_h < input_height) && (input_w >=0) && (input_w < input_width)) {
                  scalar_t val = weight[weight_offset + kh * kernel_w + kw];
                  atomicAdd(&output[n][oc][output_h][output_w], input[n][ic][input_h][input_w] * val);
              }
          }
      }
  };

  // Launch the kernel
  at::cuda::CUDA降雨: parallel_for(grid, block, 0, conv_transpose_op);

  AT_CUDA_CHECK(cudaGetLastError());
}

at::Tensor conv_transpose2d_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    int64_t stride_h, int64_t stride_w,
    int64_t padding_h, int64_t padding_w,
    int64_t output_padding_h, int64_t output_padding_w,
    int64_t groups) {

  // Calculate output shape
  auto in_channels = input.size(1);
  auto out_channels = weight.size(0)/groups;
  auto kernel_h = weight.size(2);
  auto kernel_w = weight.size(3);

  int64_t batch = input.size(0);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  int64_t output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
  int64_t output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

  auto output = at::empty({batch, out_channels, output_height, output_width}, input.options());

  AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
      using scalar_t = float;
      const auto input_acc = input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>();
      const auto weight_acc = weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>();
      auto output_acc = output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>();

      conv_transpose2d_kernelLauncher<scalar_t>(
          input_acc, weight_acc, output_acc,
          in_channels, out_channels,
          kernel_h, kernel_w,
          stride_h, stride_w,
          padding_h, padding_w,
          output_padding_h, output_padding_w,
          groups);
  }));

  return output;
}
"""

# Compile the inline CUDA code for convolution transpose
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources="",
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.output_padding = (output_padding, output_padding)
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
        self.conv_transpose = conv_transpose2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.conv_transpose.conv_transpose2d_cuda(
            x, self.weight, self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.output_padding[0], self.output_padding[1],
            self.groups
        )
        if self.bias is not None:
            # Implement bias addition here (would need another kernel or use existing)
            result += self.bias.view(1, -1, 1, 1)
        return result