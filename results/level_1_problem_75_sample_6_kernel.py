import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4> input,
    const torch::PackedTensorAccessor<scalar_t,4> weight,
    torch::PackedTensorAccessor<scalar_t,4> output,
    int out_channels,
    int in_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups) {

    const int B = blockIdx.z;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_y >= output.size(2) || out_x >= output.size(3)) {
        return;
    }

    for (int c_out_group = threadIdx.z; c_out_group < (out_channels / groups); c_out_group += blockDim.z) {
        const int c_out = c_out_group + (groups * threadIdx.z); // Assuming groups handled via thread dims
        int c_in_group = c_out_group % (in_channels / groups);
        const int c_in = c_in_group + (groups * threadIdx.z);

        scalar_t sum = 0;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Compute input coordinates
                const int in_y = (out_y - padding_h - kh * dilation_h) / stride_h;
                const int in_x = (out_x - padding_w - kw * dilation_w) / stride_w;

                // Check if in bounds for input
                if (in_y >= 0 && in_y < input.size(2) && in_x >= 0 && in_x < input.size(3)) {
                    sum += weight[c_out][kh][kw] * input[B][c_in][in_y][in_x];
                }
            }
        }
        output[B][c_out][out_y][out_x] += sum;
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight,
    int stride_h, int stride_w, int padding_h, int padding_w,
    int dilation_h, int dilation_w, int groups) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    // Compute output dimensions
    const int output_height = (input_height - 1) * stride_h - 2 * padding_h + 
        dilation_h * (kernel_h - 1) + 1;
    const int output_width = (input_width - 1) * stride_w - 2 * padding_w +
        dilation_w * (kernel_w - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
        torch::device("cuda").dtype(input.dtype()));

    dim3 threads(16, 16, 4); // Tune based on parameters
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size); // Each batch as a separate z-block

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            out_channels,
            in_channels,
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

# Compile the custom CUDA operator
conv_transpose2d_cpp = "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups);"
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1),
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride_h, self.stride_w = stride
        self.padding_h, self.padding_w = padding
        self.dilation_h, self.dilation_w = dilation
        self.groups = groups

        # Initialize weight parameters similar to PyTorch's ConvTranspose2d
        kernel_h, kernel_w = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_h, kernel_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv_transpose2d.conv_transpose2d_cuda(
            x.cuda(),
            self.weight,
            self.stride_h, self.stride_w,
            self.padding_h, self.padding_w,
            self.dilation_h, self.dilation_w,
            self.groups
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output