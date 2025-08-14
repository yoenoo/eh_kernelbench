import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int kH, int kW,
    int in_height, int in_width,
    int out_height, int out_width,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int groups) {

  const int output_size = out_height * out_width;
  const int channels_per_group = in_channels / groups;

  CUDA_KERNEL_LOOP(index, batch_size * out_channels * output_size) {
    int w = index % out_width;
    int h = (index / out_width) % out_height;
    int c_out = (index / (out_height * out_width)) % out_channels;
    int n = index / (out_channels * out_height * out_width);

    int g = c_out / (out_channels / groups);
    c_out = c_out % (out_channels / groups);

    scalar_t val = 0;
    for (int kh = 0; kh < kH; ++kh) {
      for (int kw = 0; kw < kW; ++kw) {
        // Compute the input coordinates
        const int h_in = (h - kh * stride_h - padding_h + output_padding_h) / stride_h;
        const int w_in = (w - kw * stride_w - padding_w + output_padding_w) / stride_w;

        // Check if the current kernel element contributes to this output position
        if ((h - kh * stride_h - padding_h + output_padding_h) % stride_h != 0 ||
            (w - kw * stride_w - padding_w + output_padding_w) % stride_w != 0) {
          continue;
        }

        h_in = h_in / 1;  // stride_h is considered in the condition above
        w_in = w_in / 1;  // stride_w same

        if (h_in < 0 || h_in >= in_height || w_in < 0 || w_in >= in_width) {
          continue;
        }

        for (int c_in = 0; c_in < channels_per_group; ++c_in) {
          const int input_offset = ((n * in_channels + g * channels_per_group + c_in) * in_height + h_in) * in_width + w_in;
          const int weight_offset = (g * (out_channels / groups) * kH * kW * channels_per_group +
                                    c_out * kH * kW * channels_per_group +
                                    c_in * kH * kW +
                                    kh * kW + kw);
          val += input[input_offset] * weight[weight_offset];
        }
      }
    }

    const int output_offset = ((n * out_channels + c_out) * out_height + h) * out_width + w;
    output[output_offset] = val;
  }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w,
                                   int padding_h, int padding_w, int output_padding_h, int output_padding_w, int groups) {

  const auto batch_size = input.size(0);
  const auto in_channels = input.size(1);
  const auto in_height = input.size(2);
  const auto in_width = input.size(3);

  const auto out_channels = weight.size(0);  // Assuming weight is [out_channels, in_channels/groups, kH, kW]
  const auto kH = weight.size(2);
  const auto kW = weight.size(3);

  // Output dimensions calculation based on transposed convolution formula
  const auto out_height = (in_height - 1) * stride_h - 2 * padding_h + kH + output_padding_h;
  const auto out_width = (in_width - 1) * stride_w - 2 * padding_w + kW + output_padding_w;

  auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());

  dim3 blocks(TORCH_DIV_UP(batch_size * out_channels * out_height * out_width, 512));
  dim3 threads(512);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&]{
    conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
      input.data_ptr<scalar_t>(),
      weight.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      batch_size, in_channels, out_channels,
      kH, kW,
      in_height, in_width,
      out_height, out_width,
      stride_h, stride_w,
      padding_h, padding_w,
      output_padding_h, output_padding_w,
      groups
    );
  }));

  cudaDeviceSynchronize();
  return output;
}
"""

conv_transpose2d_header = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w, int output_padding_h, int output_padding_w, int groups);
"""

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_header,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=False,
    extra_cuda_cflags=['-lineinfo'],
    extra_cflags=['-lineinfo']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Weights initialization similar to PyTorch's ConvTranspose2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Similar to PyTorch's default

        # Bias is not used here since the original model has bias=False
        # If bias were True, would need to handle it here

    def forward(self, x):
        return conv_transpose2d.conv_transpose2d_cuda(
            x,
            self.weight,
            self.stride, self.stride,
            self.padding, self.padding,
            self.output_padding, self.output_padding,
            self.groups
        )