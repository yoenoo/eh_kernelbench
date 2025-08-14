import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ weight,
                                       scalar_t* __restrict__ output,
                                       int batch_size,
                                       int in_channels,
                                       int out_channels,
                                       int input_depth,
                                       int input_height,
                                       int input_width,
                                       int kernel_size,
                                       int stride,
                                       int padding) {

  const int out_depth = (input_depth - 1) * stride - 2 * padding + kernel_size;
  const int out_height = (input_height - 1) * stride - 2 * padding + kernel_size;
  const int out_width = (input_width - 1) * stride - 2 * padding + kernel_size;

  CUDA_1D_KERNEL_LOOP(index, batch_size * out_channels * out_depth * out_height * out_width) {
    int output_offset = index;
    int w = output_offset % out_width;
    output_offset /= out_width;
    int h = output_offset % out_height;
    output_offset /= out_height;
    int d = output_offset % out_depth;
    output_offset /= out_depth;
    int c_out = output_offset % out_channels;
    int n = output_offset / out_channels;

    scalar_t val = 0;
    for (int k_z = 0; k_z < kernel_size; ++k_z) {
      for (int k_y = 0; k_y < kernel_size; ++k_y) {
        for (int k_x = 0; k_x < kernel_size; ++k_x) {
          int input_d = (d - k_z + 2 * padding) / stride;
          int input_h = (h - k_y + 2 * padding) / stride;
          int input_w = (w - k_x + 2 * padding) / stride;

          if ((input_d < 0) || (input_d >= input_depth) ||
              (input_h < 0) || (input_h >= input_height) ||
              (input_w < 0) || (input_w >= input_width)) {
            continue;
          }

          for (int c_in_group = 0; c_in_group < in_channels / groups; ++c_in_group) {
            int c_in = c_in_group + (c_out / (out_channels / groups)) * (in_channels / groups);

            int weight_offset = ((c_out * kernel_size * kernel_size * kernel_size) + 
                                k_z * kernel_size * kernel_size +
                                k_y * kernel_size +
                                k_x) * (in_channels / groups) + c_in_group;

            val += input[n * in_channels * input_depth * input_height * input_width + 
                        c_in * input_depth * input_height * input_width +
                        input_d * input_height * input_width +
                        input_h * input_width + input_w] *
                   weight[weight_offset];
          }
        }
      }
    }

    output[index] = val;
  }
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int stride,
                                   int padding,
                                   int groups) {
  const auto batch_size = input.size(0);
  const auto in_channels = input.size(1);
  const auto out_channels = weight.size(0) * groups;
  const auto input_depth = input.size(2);
  const auto input_height = input.size(3);
  const auto input_width = input.size(4);
  const auto kernel_size = weight.size(2); // assuming square kernel

  const int out_depth = (input_depth - 1) * stride - 2 * padding + kernel_size;
  const int out_height = (input_height - 1) * stride - 2 * padding + kernel_size;
  const int out_width = (input_width - 1) * stride - 2 * padding + kernel_size;

  auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width},
                            input.options());

  dim3 blocks(
      ATen::cuda::getTensorCoreBlockCount(output.numel(), 256),
      1,
      1);
  dim3 threads(256, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_cuda", ([&] {
    conv_transpose3d_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding);
  }));

  cudaDeviceSynchronize();
  return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int stride,
                                   int padding,
                                   int groups);
"""

# Compile the inline CUDA code for 3D transposed convolution
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cuda_sources=conv_transpose3d_source,
    cpp_sources=conv_transpose3d_cpp_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=["-g", "-O3"],
    extra_cuda_cflags=["-g", "-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # Weight initialization mimicking PyTorch's ConvTranspose3d initialization
        self.weight = nn.Parameter(torch.empty(groups*out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return conv_transpose3d.conv_transpose3d_cuda(x, self.weight, self.stride, self.padding, self.groups)