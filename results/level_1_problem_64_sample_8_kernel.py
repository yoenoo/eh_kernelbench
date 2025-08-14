import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv1d_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Based on the transposed convolution kernel implementation from PyTorch's cunn/generated/TransposeConvolution.cu

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int input_height,
    int output_height,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int dilation,
    int groups,
    int in_channels_per_group,
    int out_channels_per_group) {
  CUDA_1D_KERNEL_LOOP(index, output_height) {
    int out_pos = index;
    int batch_out_c = index / output_height;
    int out_channel = batch_out_c % out_channels_per_group;
    int batch = batch_out_c / out_channels_per_group;

    out_channel += out_channels_per_group * (groups * (batch / groups));
    // Output channel must be within current group
    int in_channel_group = out_channel / out_channels_per_group;
    int in_channel = in_channel_group * in_channels_per_group;

    int in_off = batch * in_channels_per_group * input_height;

    int weight_off = out_channel * kernel_size * in_channels_per_group;

    int effective_kernel_size = (kernel_size - 1) * dilation + 1;

    int output_start = out_pos - padding;
    int input_start = (output_start) / stride;
    if (output_start % stride != 0) {
      continue;
    }
    int input_pos = input_start;
    int weight_pos = 0;

    // Iterate over the kernel
    for (int w = 0; w < kernel_size; ++w) {
      int current_out_pos = output_start + dilation * w;
      if (current_out_pos < 0 || current_out_pos >= output_height) {
        continue;
      }

      int current_in_pos = current_out_pos / stride;
      if (current_out_pos % stride != 0) {
        continue;
      }

      if (current_in_pos < 0 || current_in_pos >= input_height) {
        continue;
      }

      for (int c = 0; c < in_channels_per_group; ++c) {
        int wgt_idx = weight_off + (c * kernel_size + w);
        int inp_idx = in_off + (in_channel + c) * input_height + current_in_pos;
        int out_idx = batch * out_channels_per_group * output_height +
                      out_channel * output_height + current_out_pos;
        atomicAdd(output + out_idx, static_cast<scalar_t>(input[inp_idx] * weight[wgt_idx]));
      }
    }
  }
}

at::Tensor conv_transpose1d_cuda(
    const at::Tensor &input,
    const at::Tensor &weight,
    int stride,
    int padding,
    int output_padding,
    int groups) {
  const int batch = input.size(0);
  const int in_channels = input.size(1);
  const int input_height = input.size(2);
  const int out_channels = weight.size(0);
  const int kernel_size = weight.size(2);

  int out_channels_per_group = out_channels / groups;
  int in_channels_per_group = in_channels / groups;

  // Output size calculation
  int output_height = (input_height - 1) * stride - 2 * padding +
                      kernel_size + output_padding;

  auto output = at::zeros({batch, out_channels, output_height}, input.options());

  const int num_threads = 1024;
  int elements = batch * out_channels * output_height;
  int blocks = (elements + num_threads - 1) / num_threads;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose1d_cuda", ([&] {
    conv_transpose1d_kernel<scalar_t><<<blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data<scalar_t>(),
        weight.data<scalar_t>(),
        output.data<scalar_t>(),
        input_height,
        output_height,
        kernel_size,
        stride,
        padding,
        output_padding,
        /*dilation=*/1,  // Assuming dilation is fixed to 1 for simplicity
        groups,
        in_channels_per_group,
        out_channels_per_group);
  }));

  return output;
}
"""

conv1d_transpose_cpp_source = (
    "at::Tensor conv_transpose1d_cuda(const at::Tensor &input, const at::Tensor &weight, int stride, int padding, int output_padding, int groups);"
)

conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=conv1d_transpose_cpp_source,
    cuda_sources=conv1d_transpose_source,
    functions=["conv_transpose1d_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA", "-x cuda"],
    extra_cuda_cflags=["--expt-extended-lambda"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        self.bias_term = nn.Parameter(torch.empty(out_channels), requires_grad=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_term, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv_transpose1d.conv_transpose1d_cuda(
            x, 
            self.weight, 
            self.stride, 
            self.padding, 
            self.output_padding, 
            self.groups
        )
        if self.bias:
            output += self.bias_term.view(1, -1, 1)
        return output