import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized ConvTranspose2d
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(scalar_t* __restrict__ output,
                                       const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ weight,
                                       const scalar_t* __restrict__ bias,
                                       int batch_size,
                                       int in_channels,
                                       int out_channels,
                                       int kernel_size,
                                       int height_in,
                                       int width_in,
                                       int height_out,
                                       int width_out,
                                       int stride,
                                       int padding,
                                       int dilation) {

    const int output_size = batch_size * out_channels * height_out * width_out;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    int output_depth = out_channels;
    int output_height = height_out;
    int output_width = width_out;

    int output_channel = idx / (output_height * output_width);
    int residual = idx % (output_height * output_width);
    int output_row = residual / output_width;
    int output_col = residual % output_width;

    scalar_t value = 0;
    for (int input_channel = 0; input_channel < in_channels; ++input_channel) {
        for (int kernel_row = 0; kernel_row < kernel_size; ++kernel_row) {
            for (int kernel_col = 0; kernel_col < kernel_size; ++kernel_col) {
                // Compute input coordinates
                int input_row = output_row * stride - padding + (kernel_row) * dilation;
                int input_col = output_col * stride - padding + (kernel_col) * dilation;
                // Check if within input bounds
                if (input_row >= 0 && input_row < height_in && input_col >= 0 && input_col < width_in) {
                    int kernel_idx = kernel_row * kernel_size + kernel_col;
                    int weight_offset = (output_channel * in_channels + input_channel) * kernel_size * kernel_size + kernel_idx;
                    int input_offset = (input_channel * height_in + input_row) * width_in + input_col;
                    value += weight[weight_offset] * input[input_offset];
                }
            }
        }
    }

    if (bias != nullptr) {
        value += bias[output_channel];
    }

    // Calculate the output index
    int output_offset = (output_channel * height_out + output_row) * width_out + output_col;
    output[output_offset] = value;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv_transpose2d_cuda(torch::Tensor input,
                                                                 torch::Tensor weight,
                                                                 torch::Tensor bias,
                                                                 int stride,
                                                                 int padding,
                                                                 int dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height_in = input.size(2);
    const int width_in = input.size(3);
    const int out_channels = weight.size(0); // weight is [out_channels, in_channels, kernel_size, kernel_size]
    const int kernel_size = weight.size(2);
    
    // Compute output dimensions
    const int height_out = (height_in - 1) * stride - 2 * padding + kernel_size + 2 * padding;
    const int width_out = (width_in - 1) * stride - 2 * padding + kernel_size + 2 * padding;

    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());
    auto output_reshaped = output.contiguous().view({-1});

    int threads = 256;
    int blocks = (output.numel() + threads - 1) / threads;

    const int kernel_size_val = kernel_size;
    const int stride_val = stride;
    const int padding_val = padding;
    const int dilation_val = dilation;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            output_reshaped.data_ptr<scalar_t>(),
            input.contiguous().data_ptr<scalar_t>(),
            weight.contiguous().data_ptr<scalar_t>(),
            (bias.defined() ? bias.data_ptr<scalar_t>() : nullptr),
            batch_size,
            in_channels,
            out_channels,
            kernel_size_val,
            height_in,
            width_in,
            height_out,
            width_out,
            stride_val,
            padding_val,
            dilation_val);
    }));

    cudaDeviceSynchronize();
    return std::make_tuple(output, weight, bias);
}
"""

cpp_source = """
#include <vector>
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv_transpose2d_cuda(torch::Tensor input,
                                                                 torch::Tensor weight,
                                                                 torch::Tensor bias,
                                                                 int stride,
                                                                 int padding,
                                                                 int dilation);
"""

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Load the custom CUDA kernel
        self.custom_conv = load_inline(
            name='conv_transpose2d',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=['conv_transpose2d_cuda'],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        bias_tensor = self.bias if self.bias is not None else torch.Tensor()
        output, _, _ = self.custom_conv.conv_transpose2d_cuda(
            x,
            self.weight,
            bias_tensor,
            self.stride,
            self.padding,
            self.dilation
        )
        return output.view(x.size(0), -1, output.size(2), output.size(3))