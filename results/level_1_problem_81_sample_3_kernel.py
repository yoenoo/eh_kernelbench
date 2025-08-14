import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2D
conv_transpose2d_cuda = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ weight,
                                       scalar_t* __restrict__ output,
                                       const int batch_size,
                                       const int in_channels,
                                       const int out_channels,
                                       const int kernel_h,
                                       const int kernel_w,
                                       const int stride,
                                       const int padding,
                                       const int dilation,
                                       const int input_h,
                                       const int input_w,
                                       const int output_h,
                                       const int output_w) {
    const int batch = blockIdx.x;
    const int out_channel = blockIdx.y;
    const int out_y = blockIdx.z * blockDim.y + threadIdx.y;
    const int out_x = threadIdx.x;

    if (out_y >= output_h || out_x >= output_w) return;

    for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
        // Compute the input coordinates based on transposed convolution
        const int in_y = (out_y - padding) / stride;
        const int in_x = (out_x - padding) / stride;

        // Determine the effective kernel position based on the output coordinates
        const int rel_y = out_y - stride * in_y - padding;
        const int rel_x = out_x - stride * in_x - padding;

        // Ensure that the kernel position is within the kernel dimensions
        if (rel_y % dilation != 0 || rel_x % dilation != 0) continue;
        const int k_y = rel_y / dilation;
        const int k_x = rel_x / dilation;

        if (k_y < 0 || k_y >= kernel_h || k_x < 0 || k_x >= kernel_w) continue;
        if (in_y < 0 || in_y >= input_h || in_x < 0 || in_x >= input_w) continue;

        const int weight_index = (in_channel * out_channels + out_channel) * kernel_h * kernel_w + k_y * kernel_w + k_x;
        const int input_offset = (batch * in_channels + in_channel) * input_h * input_w + in_y * input_w + in_x;
        const int output_offset = (batch * out_channels + out_channel) * output_h * output_w + out_y * output_w + out_x;

        atomicAdd(&output[output_offset], input[input_offset] * weight[weight_index]);
    }
}

std::tuple<torch::Tensor> conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight,
    int stride, int padding, int dilation) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_h = input.size(2);
    const auto input_w = input.size(3);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);
    const auto out_channels = weight.size(0);

    // Compute output dimensions
    const int output_h = (input_h - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + 1;
    const int output_w = (input_w - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());

    const dim3 threads(32, 8); // Tuned thread block size (X, Y)
    dim3 blocks(batch_size, out_channels, (output_h + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            kernel_h, kernel_w,
            stride, padding, dilation,
            input_h, input_w,
            output_h, output_w
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

# Compile the CUDA code
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources="",
    cuda_sources=conv_transpose_cuda,
    functions=['conv_transpose2d_cuda'],
    verbose=True,
    extra_cuda_cflags=['-lineinfo']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        # Initialize parameters similar to nn.ConvTranspose2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv_transpose2d = conv_transpose2d

    def forward(self, x):
        output = self.conv_transpose2d.conv_transpose2d_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output