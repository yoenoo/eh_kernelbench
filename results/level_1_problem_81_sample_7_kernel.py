import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

# CUDA Kernel Code for Transposed Convolution
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_size,
    const int height_in,
    const int width_in,
    const int height_out,
    const int width_out,
    const int stride,
    const int padding,
    const int dilation
) {
    // Compute the output coordinates
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;

    if (h_out >= height_out || w_out >= width_out) return;

    // Iterate over input channels and kernel elements
    for (int cin = 0; cin < in_channels; ++cin) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Compute input coordinates considering dilation and padding
                int h_in = (h_out - padding - kh * dilation) / stride;
                int w_in = (w_out - padding - kw * dilation) / stride;

                // Check if input coordinates are valid
                if (h_in < 0 || h_in >= height_in || w_in < 0 || w_in >= width_in) continue;

                // Compute the output increment
                for (int cout = 0; cout < out_channels; ++cout) {
                    const int input_offset = 
                        batch * in_channels * height_in * width_in +
                        cin * height_in * width_in +
                        h_in * width_in + w_in;

                    const int weight_offset = 
                        cout * in_channels * kernel_size * kernel_size +
                        cin * kernel_size * kernel_size +
                        kh * kernel_size + kw;

                    const int output_offset = 
                        batch * out_channels * height_out * width_out +
                        cout * height_out * width_out +
                        h_out * width_out + w_out;

                    atomicAdd(output + output_offset, 
                             input[input_offset] * weight[weight_offset]);
                }
            }
        }
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int height_in = input.size(2);
    const int width_in = input.size(3);

    // Compute output dimensions based on transposed convolution formula
    const int height_out = (height_in - 1) * stride - 2 * padding + kernel_size + 2 * padding;
    const int width_out = (width_in - 1) * stride - 2 * padding + kernel_size + 2 * padding;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({batch_size, out_channels, height_out, width_out}, output_options);

    dim3 threads(16, 16); // Tune based on hardware
    dim3 blocks(
        (width_out + threads.x - 1) / threads.x,
        (height_out + threads.y - 1) / threads.y,
        batch_size
    );

    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        height_in,
        width_in,
        height_out,
        width_out,
        stride,
        padding,
        dilation
    );

    cudaDeviceSynchronize();
    return output;
}
"""

# Load the CUDA extension
conv_transpose_module = load(
    name="conv_transpose2d_cuda",
    sources=[cuda_source],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv_transpose_module.conv_transpose2d_cuda(
            x, self.weight, self.stride, self.padding, self.dilation
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output