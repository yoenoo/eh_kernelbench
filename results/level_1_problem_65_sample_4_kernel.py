import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int groups
) {
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z / out_channels;
    const int out_channel = blockIdx.z % out_channels;

    if (out_x >= output_width || out_y >= output_height || batch >= batch_size) {
        return;
    }

    const int in_channel_group = out_channel / (out_channels / groups);
    const int group_id = in_channel_group % groups;
    const int in_channel = in_channel_group % (in_channels / groups);

    float sum = 0.0;
    for (int kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
        for (int kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
            // Compute the corresponding input coordinates
            const int in_y = (out_y - kernel_y - padding_h) / stride_h;
            const int in_x = (out_x - kernel_x - padding_w) / stride_w;

            // Check if the input coordinates are valid
            if ((out_y - kernel_y - padding_h) % stride_h != 0 ||
                (out_x - kernel_x - padding_w) % stride_w != 0 ||
                in_y < 0 || in_y >= input_height ||
                in_x < 0 || in_x >= input_width) {
                continue;
            }

            const int input_offset = batch * in_channels * input_height * input_width
                                   + (group_id * (in_channels / groups) + in_channel) * input_height * input_width
                                   + in_y * input_width + in_x;
            const int weight_offset = group_id * (out_channels / groups) * kernel_h * kernel_w * (in_channels / groups)
                                    + (out_channel % (out_channels / groups)) * kernel_h * kernel_w * (in_channels / groups)
                                    + kernel_y * kernel_w * (in_channels / groups)
                                    + kernel_x * (in_channels / groups)
                                    + in_channel % (in_channels / groups);
            sum += input[input_offset] * weight[weight_offset];
        }
    }

    const int output_offset = batch * out_channels * output_height * output_width
                           + out_channel * output_height * output_width
                           + out_y * output_width + out_x;
    output[output_offset] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int groups
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0) / groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    const int output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    torch::Tensor output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 threads(16, 16); // Tunable thread block size
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * out_channels
    );

    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        input_height,
        input_width,
        output_height,
        output_width,
        groups
    );

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w, int output_padding_h, int output_padding_w, int groups);"
)

# Compile the custom kernel
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.bias = bias

        # Initialize weights similar to PyTorch's ConvTranspose2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        output_padding_h, output_padding_w = self.output_padding

        # Run custom CUDA kernel
        output = conv_transpose2d.conv_transpose2d_cuda(
            x,
            self.weight,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_padding_h,
            output_padding_w,
            self.groups
        )

        if self.bias is not None:
            # Add bias (can be merged into the kernel if performance critical)
            output += self.bias.view(1, -1, 1, 1)

        return output