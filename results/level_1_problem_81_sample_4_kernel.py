import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d
conv_transpose_2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(const torch::PackedTensorAccessor<scalar_t,4> input,
                                       const torch::PackedTensorAccessor<scalar_t,4> weight,
                                       torch::PackedTensorAccessor<scalar_t,4> output,
                                       const int out_channels,
                                       const int in_channels,
                                       const int kernel_h,
                                       const int kernel_w,
                                       const int stride_h,
                                       const int stride_w,
                                       const int padding_h,
                                       const int padding_w,
                                       const int dilation_h,
                                       const int dilation_w) {
    const int n = blockIdx.z;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_y >= output.size(2) || out_x >= output.size(3)) {
        return;
    }

    for (int c_out = 0; c_out < out_channels; ++c_out) {
        scalar_t val = 0;
        for (int k_h = 0; k_h < kernel_h; ++k_h) {
            for (int k_w = 0; k_w < kernel_w; ++k_w) {
                const int in_y = (out_y - padding_h - dilation_h * k_h) / stride_h;
                const int in_x = (out_x - padding_w - dilation_w * k_w) / stride_w;
                
                // Check input boundaries
                if (in_y < 0 || in_y >= input.size(2) || in_x < 0 || in_x >= input.size(3)) {
                    continue;
                }
                
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    val += weight[c_out][c_in][k_h][k_w] * input[n][c_in][in_y][in_x];
                }
            }
        }
        output[n][c_out][out_y][out_x] = val;
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                    torch::Tensor weight,
                                    int stride_h,
                                    int stride_w,
                                    int padding_h,
                                    int padding_w,
                                    int dilation_h,
                                    int dilation_w) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);
    const auto out_channels = weight.size(0);
    
    // Compute output dimensions
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);
    const auto output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1;
    const auto output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());
    
    dim3 threads(16, 16);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size
    );

    const int CUDA_tpb = threads.x * threads.y;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w
        );
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

# Corresponding C++ headers
conv_transpose_2d_cpp_source = (
    "#include <torch/extension.h>\n"
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);"
)

# Load the CUDA extensions
conv_transpose_module = load_inline(
    name='conv_transpose2d',
    cpp_sources=conv_transpose_2d_cpp_source,
    cuda_sources=conv_transpose_2d_source,
    functions=['conv_transpose2d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        self.cuda_conv_transpose = conv_transpose_module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.cuda_conv_transpose.conv_transpose2d_cuda(
            x,
            self.weight,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output