import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_3D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

// Custom 3D transpose convolution forward kernel
template <typename scalar_t>
__global__ void ConvTranspose3dForwardKernel(
    const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> output,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t dilation) {

  // Define the output dimensions
  int64_t batch = output.size(0);
  int64_t out_depth = output.size(2);
  int64_t out_height = output.size(3);
  int64_t out_width = output.size(4);

  CUDA_3D_KERNEL_LOOP(index, batch * out_channels * out_depth * out_height * out_width) {
    int w = index % out_width;
    int h = (index / out_width) % out_height;
    int d = (index / (out_width * out_height)) % out_depth;
    int c = (index / (out_width * out_height * out_depth)) % out_channels;
    int n = index / (out_channels * out_depth * out_height * out_width);

    scalar_t val = 0;
    // Iterate over kernel and input
    for (int kd = 0; kd < kernel_size; ++kd) {
      for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
          // Compute the input coordinates
          int input_d = (d - kd - padding) / stride;
          int input_h = (h - kh - padding) / stride;
          int input_w = (w - kw - padding) / stride;

          if (input_d < 0 || input_h < 0 || input_w < 0 ||
              input_d >= input.size(2) || input_h >= input.size(3) || input_w >= input.size(4)) {
            continue;
          }

          // Compute channel index in input
          for (int ic = 0; ic < in_channels; ++ic) {
              // Access weight (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
              val += weight[c][ic][kd][kh][kw] * input[n][ic][input_d][input_h][input_w];
          }
        }
      }
    }
    output[n][c][d][h][w] = val;
  }
}

torch::Tensor conv_transpose3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t dilation) {

  // Output shape calculation
  auto input_size = input.sizes();
  int64_t batch = input_size[0];
  int64_t in_channels = input_size[1];
  int64_t in_depth = input_size[2];
  int64_t in_height = input_size[3];
  int64_t in_width = input_size[4];

  auto weight_size = weight.sizes();
  int64_t out_channels = weight_size[1]; // Since weight is [in_channels, out_channels, ...]?
  // Correction: Actually, weight should be [out_channels, in_channels, kernel_size, kernel_size, kernel_size]
  // Thus, out_channels = weight.size(0), in_channels = weight.size(1)
  out_channels = weight.size(0);
  in_channels = weight.size(1);
  int64_t kernel_size = weight.size(2);

  // Compute output dimensions (assuming stride, padding, etc. as parameters, this is a simplified example)
  int64_t out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
  int64_t out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
  int64_t out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

  auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
  auto output = torch::zeros({batch, out_channels, out_depth, out_height, out_width}, output_options);

  dim3 blocks;
  dim3 threads(256);
  int64_t num_elements = batch * out_channels * out_depth * out_height * out_width;
  blocks.x = (num_elements + threads.x - 1) / threads.x;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_forward_cuda", ([&] {
    ConvTranspose3dForwardKernel<scalar_t><<<blocks, threads>>>(
      input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
      weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
      output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
      in_channels,
      out_channels,
      kernel_size,
      stride,
      padding,
      output_padding,
      dilation);
  }));

  cudaDeviceSynchronize();
  return output;
}
"""

conv_transpose3d_cpp_source = """
#include <torch/extension.h>
torch::Tensor conv_transpose3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t dilation);
"""

# Load the CUDA extension
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=[conv_transpose3d_cpp_source],
    cuda_sources=[conv_transpose3d_source],
    functions=["conv_transpose3d_forward_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, 
                 output_padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.kernel_size = kernel_size

        # Initialize weights similar to PyTorch's ConvTranspose3d
        kernel_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(kernel_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Example initialization

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv_transpose3d.conv_transpose3d_forward_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation
        )

        # Handle bias addition if needed
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)

        return output