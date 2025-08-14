import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA Conv2D implementation
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void simple_conv2d_forward(const float* __restrict__ input,
                                      const float* __restrict__ weight,
                                      float* __restrict__ output,
                                      int batch_size,
                                      int in_channels,
                                      int out_channels,
                                      int kernel_size,
                                      int input_height,
                                      int input_width,
                                      int output_height,
                                      int output_width,
                                      int stride,
                                      int padding,
                                      int dilation) {

    int H_out = output_height;
    int W_out = output_width;
    
    int batch_idx = blockIdx.x;
    int out_channel = blockIdx.y;
    int out_row = threadIdx.y;
    int out_col = threadIdx.x;

    float acc = 0.0;
    for (int kernel_row = 0; kernel_row < kernel_size; ++kernel_row) {
        for (int kernel_col = 0; kernel_col < kernel_size; ++kernel_col) {
            for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
                // Compute input spatial coordinates
                int in_row = out_row * stride + kernel_row * dilation - padding;
                int in_col = out_col * stride + kernel_col * dilation - padding;

                // Skip input positions outside the image
                if (in_row >= 0 && in_row < input_height &&
                    in_col >= 0 && in_col < input_width) {
                    acc += input[batch_idx * in_channels * input_height * input_width +
                                in_channel * input_height * input_width +
                                in_row * input_width + in_col] *
                           weight[out_channel * in_channels * kernel_size * kernel_size +
                                  in_channel * kernel_size * kernel_size +
                                  kernel_row * kernel_size + kernel_col];
                }
            }
        }
    }

    // Write to output
    output[batch_idx * out_channels * H_out * W_out +
           out_channel * H_out * W_out +
           out_row * W_out + out_col] = acc;
}

torch::Tensor simple_conv2d(torch::Tensor input,
                           torch::Tensor weight,
                           int stride,
                           int padding,
                           int dilation) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    // Compute output dimensions
    const int output_height = (input_height + 2 * padding - 
                            dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - 
                            dilation * (kernel_size - 1) - 1) / stride + 1;

    // Output tensor
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                              input.options());

    dim3 threads(output_width, output_height);
    dim3 blocks(batch_size, out_channels);

    simple_conv2d_forward<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        input_height,
        input_width,
        output_height,
        output_width,
        stride,
        padding,
        dilation
    );

    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor simple_conv2d(torch::Tensor input,
                           torch::Tensor weight,
                           int stride,
                           int padding,
                           int dilation);
"""

# Compile the CUDA extension
simple_conv2d = load_inline(
    name='simple_conv2d',
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=['simple_conv2d'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights with same shape as PyTorch Conv2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        
        # Note: This simple implementation doesn't handle groups/bias
        if groups != 1 or bias:
            raise NotImplementedError("Groups and bias not yet implemented in this basic example")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return simple_conv2d.simple_conv2d(x, self.weight, self.stride, self.padding, self.dilation)