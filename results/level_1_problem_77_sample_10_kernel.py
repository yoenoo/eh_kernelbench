import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void ConvTranspose3DForwardKernel(
    const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> output,
    int in_channels, int out_channels, int kernel_size,
    int stride, int padding, int dilation,
    int batch_size, int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int kernel_depth, int kernel_height, int kernel_width) {

  CUDA_1D_KERNEL_LOOP(output_idx, batch_size * out_channels * out_depth * out_height * out_width) {
    int w = output_idx % out_width;
    int h = (output_idx / out_width) % out_height;
    int d = (output_idx / (out_width * out_height)) % out_depth;
    int c = (output_idx / (out_width * out_height * out_depth)) % out_channels;
    int n = output_idx / (out_channels * out_depth * out_height * out_width);

    scalar_t val = 0;
    for (int kd = 0; kd < kernel_depth; ++kd) {
      for (int kh = 0; kh < kernel_height; ++kh) {
        for (int kw = 0; kw < kernel_width; ++kw) {
          for (int ic = 0; ic < in_channels; ++ic) {
            // Compute input indices
            int id = (d - kd * dilation) / stride;
            int ih = (h - kh * dilation) / stride;
            int iw = (w - kw * dilation) / stride;

            // Check if the input indices are within bounds
            if (id < 0 || id >= in_depth || ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) {
              continue;
            }

            // Check if the current output position is covered by the input and kernel
            if ((d - kd * dilation) % stride != 0 ||
                (h - kh * dilation) % stride != 0 ||
                (w - kw * dilation) % stride != 0) {
              continue;
            }

            val += input[n][ic][id][ih][iw] * weight[ic][c][kd][kh][kw];
          }
        }
      }
    }
    output[n][c][d][h][w] = val;
  }
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int stride, int padding, int dilation) {
    auto output_options = torch::TensorOptions().like(input);
    auto in_channels = input.size(1);
    auto batch_size = input.size(0);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);

    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);

    // Compute output dimensions
    auto out_depth = (in_depth - 1) * stride - 2 * padding + kernel_depth + 2 * padding;
    auto out_height = (in_height - 1) * stride - 2 * padding + kernel_height + 2 * padding;
    auto out_width = (in_width - 1) * stride - 2 * padding + kernel_width + 2 * padding;

    // Adjust based on dilation, need to recalculate properly
    out_depth = (in_depth - 1) * stride + (kernel_depth - 1) * dilation + 1 - 2 * padding;
    out_height = (in_height - 1) * stride + (kernel_height - 1) * dilation + 1 - 2 * padding;
    out_width = (in_width - 1) * stride + (kernel_width - 1) * dilation + 1 - 2 * padding;

    auto output = torch::zeros({batch_size, weight.size(1), out_depth, out_height, out_width}, output_options);

    dim3 threads(256);
    dim3 blocks((batch_size * output.size(1) * out_depth * out_height * out_width + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_cuda", ([&] {
      ConvTranspose3DForwardKernel<scalar_t><<<blocks, threads>>>(
          input.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits>(),
          weight.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits>(),
          output.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits>(),
          in_channels, weight.size(1), kernel_width,
          stride, padding, dilation,
          batch_size, in_depth, in_height, in_width,
          out_depth, out_height, out_width,
          kernel_depth, kernel_height, kernel_width);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose3d_cpp_source = (
    "torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"
)

# Compile the inline CUDA code for 3D transposed convolution
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=["-D_ENABLE_CUDA.innerparsers=1"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv_transpose3d.conv_transpose3d_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            # Add bias here (might need another kernel for bias addition)
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output