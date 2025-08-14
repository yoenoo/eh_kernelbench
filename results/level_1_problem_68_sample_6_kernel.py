import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <utility>

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const torch::PackedTensorAccessor<scalar_t,5> input,
    const torch::PackedTensorAccessor<scalar_t,5> weight,
    torch::PackedTensorAccessor<scalar_t,5> output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth_in,
    int width_in,
    int height_in,
    int kernel_depth,
    int kernel_width,
    int kernel_height,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w) {

    const int out_depth = output.dims()[2];
    const int out_height = output.dims()[3];
    const int out_width = output.dims()[4];

    const int di = blockIdx.z * blockDim.z + threadIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (di >= out_depth || y >= out_height || x >= out_width) {
        return;
    }

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int out_ch = 0; out_ch < out_channels; ++out_ch) {
            scalar_t val = 0;
            for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                for (int kd = 0; kd < kernel_depth; ++kd) {
                    for (int kh = 0; kh < kernel_height; ++kh) {
                        for (int kw = 0; kw < kernel_width; ++kw) {
                            const int d_in = (di - kd * stride_d - output_padding_d + padding_d) / stride_d;
                            const int h_in = (y - kh * stride_h - output_padding_h + padding_h) / stride_h;
                            const int w_in = (x - kw * stride_w - output_padding_w + padding_w) / stride_w;

                            if (d_in < 0 || d_in >= depth_in || h_in < 0 || h_in >= height_in || w_in < 0 || w_in >= width_in) {
                                continue;
                            }

                            val += input[batch][in_ch][d_in][h_in][w_in] * 
                                   weight[out_ch][in_ch][kd][kh][kw];
                        }
                    }
                }
            }
            output[batch][out_ch][di][y][x] = val;
        }
    }
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int kernel_depth, int kernel_width, int kernel_height,
                                   int stride_d, int stride_h, int stride_w,
                                   int padding_d, int padding_h, int padding_w,
                                   int output_padding_d, int output_padding_h, int output_padding_w) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto depth_in = input.size(2);
    const auto height_in = input.size(3);
    const auto width_in = input.size(4);
    const auto out_channels = weight.size(0);

    // Calculate output dimensions
    const int out_depth = (depth_in - 1) * stride_d - 2 * padding_d + kernel_depth + output_padding_d;
    const int out_height = (height_in - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;
    const int out_width = (width_in - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, output_options);

    dim3 threads(8, 8, 8);
    dim3 blocks((out_width + threads.x - 1) / threads.x,
               (out_height + threads.y - 1) / threads.y,
               (out_depth + threads.z - 1) / threads.z);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_cuda", ([&] {
        conv_transpose3d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5>(),
            weight.packed_accessor<scalar_t,5>(),
            output.packed_accessor<scalar_t,5>(),
            batch_size,
            in_channels,
            out_channels,
            depth_in,
            width_in,
            height_in,
            kernel_depth,
            kernel_width,
            kernel_height,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            output_padding_d,
            output_padding_h,
            output_padding_w);
    }));

    return output;
}
"""

conv_transpose3d_cuda_cpp_source = """
torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int kernel_depth, int kernel_width, int kernel_height,
                                   int stride_d, int stride_h, int stride_w,
                                   int padding_d, int padding_h, int padding_w,
                                   int output_padding_d, int output_padding_h, int output_padding_w);
"""

conv_transpose3d = load_inline(
    name='conv_transpose3d',
    cpp_sources=conv_transpose3d_cuda_cpp_source,
    cuda_sources=conv_transpose3d_cuda_source,
    functions=['conv_transpose3d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), 
                 padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1, bias=False):
        super(ModelNew, self).__init__()
        
        kernel_depth, kernel_width, kernel_height = kernel_size
        stride_d, stride_h, stride_w = stride
        padding_d, padding_h, padding_w = padding
        output_padding_d, output_padding_h, output_padding_w = output_padding
        
        # Initialize weight parameters (simplified for example)
        weight_size = (out_channels, in_channels, kernel_depth, kernel_width, kernel_height)
        self.weight = nn.Parameter(torch.randn(weight_size))
        
    def forward(self, x):
        return conv_transpose3d.conv_transpose3d_cuda(
            x, 
            self.weight, 
            kernel_depth=self.weight.size(2), kernel_width=self.weight.size(3), kernel_height=self.weight.size(4),
            stride_d=1, stride_h=1, stride_w=1,
            padding_d=0, padding_h=0, padding_w=0,
            output_padding_d=0, output_padding_h=0, output_padding_w=0
        )