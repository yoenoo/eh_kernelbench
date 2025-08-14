import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2D
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                  \
      i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(const scalar_t* __restrict__ input,
                                      const scalar_t* __restrict__ weight,
                                      scalar_t* __restrict__ output,
                                      int batch, int in_channels, int out_channels,
                                      int input_height, int input_width,
                                      int kernel_h, int kernel_w,
                                      int stride_h, int stride_w,
                                      int padding_h, int padding_w) {

    CUDA_1D_KERNEL_LOOP(output_idx, batch * out_channels * input_height * input_width) {
        // Compute output indices
        int w_out = output_idx % input_width;
        int h_idx = output_idx / input_width;
        int h_out = h_idx % input_height;
        int channel_out = h_idx / input_height;
        int n = channel_out / out_channels;
        channel_out = channel_out % out_channels;

        scalar_t val = 0;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = -padding_h + h_out * stride_h - kh;
                int w_in = -padding_w + w_out * stride_w - kw;
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        int weight_idx = (channel_out * in_channels + c_in) * kernel_h * kernel_w + kh * kernel_w + kw;
                        int input_idx = n * in_channels * input_height * input_width + 
                                       c_in * input_height * input_width + 
                                       h_in * input_width + w_in;
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[output_idx] = val;
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int stride_h, int stride_w, int padding_h, int padding_w) {
    
    const auto batch = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    // Output dimensions based on transposed convolution formula
    auto output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_h;
    auto output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_w;
    
    auto output = torch::zeros({batch, out_channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int elements = batch * out_channels * output_height * output_width;
    const int blocks = (elements + threads - 1) / threads;

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&]{
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch, in_channels, out_channels,
            input_height, input_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w);
    }));

    return output;
}
"""

conv_transpose2d_cpp = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int stride_h, int stride_w, int padding_h, int padding_w);
"""

conv_transpose2d_op = load_inline(
    name='conv_transpose2d',
    cpp_sources=conv_transpose2d_cpp,
    cuda_sources=conv_transpose2d_source,
    functions=['conv_transpose2d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        # Initialize weight similar to PyTorch's ConvTranspose2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # He initialization
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv_transpose2d_op.conv_transpose2d_cuda(
            x, self.weight, self.stride[0], self.stride[1], self.padding[0], self.padding[1]
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output