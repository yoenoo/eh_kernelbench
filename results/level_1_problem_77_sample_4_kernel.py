import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

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
                                    int padding,
                                    int dilation,
                                    int output_depth,
                                    int output_height,
                                    int output_width) {
    CUDA_1D_KERNEL_LOOP(index, batch_size * out_channels * output_depth * output_height * output_width) {
        int w_idx = index % output_width;
        int h_idx = (index / output_width) % output_height;
        int d_idx = (index / (output_width * output_height)) % output_depth;
        int c_out = (index / (output_width * output_height * output_depth)) % out_channels;
        int n = index / (out_channels * output_depth * output_height * output_width);

        scalar_t val = 0;
        for (int k_depth = 0; k_depth < kernel_size; ++k_depth) {
            for (int k_height = 0; k_height < kernel_size; ++k_height) {
                for (int k_width = 0; k_width < kernel_size; ++k_width) {
                    // Compute the input coordinates
                    int input_d = d_idx - dilation * k_depth;
                    int input_h = h_idx - dilation * k_height;
                    int input_w = w_idx - dilation * k_width;

                    // Apply padding and stride
                    input_d = (input_d + stride) / stride - padding;
                    input_h = (input_h + stride) / stride - padding;
                    input_w = (input_w + stride) / stride - padding;

                    // Check boundaries
                    if (input_d < 0 || input_d >= input_depth ||
                        input_h < 0 || input_h >= input_height ||
                        input_w < 0 || input_w >= input_width) {
                        continue;
                    }

                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        val += input[n * in_channels * input_depth * input_height * input_width +
                                    c_in * input_depth * input_height * input_width +
                                    input_d * input_height * input_width +
                                    input_h * input_width +
                                    input_w] *
                               weight[c_out * in_channels * kernel_size*kernel_size*kernel_size +
                                      c_in * kernel_size*kernel_size*kernel_size +
                                      k_depth * kernel_size*kernel_size +
                                      k_height * kernel_size +
                                      k_width];
                    }
                }
            }
        }
        output[index] = val;
    }
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input,
                                  torch::Tensor weight,
                                  int kernel_size,
                                  int stride,
                                  int padding,
                                  int dilation) {
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    int out_channels = weight.size(0);
    
    // Compute output shape
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + 2 * padding;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + 2 * padding;
    int output_width = (input_height - 1) * stride - 2 * padding + kernel_size + 2 * padding;

    // Output tensor initialization
    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    int threads = 256;
    int elements = output.numel();
    int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_cuda", ([&] {
        conv_transpose3d_kernel<scalar_t><<<blocks, threads>>>(
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
            padding,
            dilation,
            output_depth,
            output_height,
            output_width);
    }));

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(torch::Tensor input,
                                  torch::Tensor weight,
                                  int kernel_size,
                                  int stride,
                                  int padding,
                                  int dilation);
"""

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Load the CUDA kernel
        self.custom_conv_transpose3d = conv_transpose3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.custom_conv_transpose3d.conv_transpose3d_cuda(
            x,
            self.weight,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        return output