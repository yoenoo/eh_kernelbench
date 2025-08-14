import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for 3D transpose convolution
conv_transpose3d_kernel = """
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(const torch::PackedTensorAccessor<scalar_t,5,const>::type input,
                            torch::PackedTensorAccessor<scalar_t,5>::type output,
                            torch::PackedTensorAccessor<scalar_t,5,const>::type weight,
                            int out_channels,
                            int in_channels,
                            int kernel_d,
                            int kernel_h,
                            int kernel_w,
                            int stride_d,
                            int stride_h,
                            int stride_w,
                            int padding_d,
                            int padding_h,
                            int padding_w,
                            int output_padding_d,
                            int output_padding_h,
                            int output_padding_w,
                            int groups) {
    const int batch_idx = blockIdx.x;
    const int out_d = blockIdx.y;
    const int out_h = blockIdx.z;
    const int out_w = threadIdx.x;

    // Compute output index within groups
    const int g = blockIdx.y / (output.size(2) / groups);
    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;

    const int group_out_d = out_d % (output.size(2)/groups);
    const int group_out_h = out_h;
    const int group_out_w = out_w;

    // Compute input coordinates
    const int in_d = (group_out_d - padding_d - output_padding_d) / stride_d;
    const int in_h = (group_out_h - padding_h - output_padding_h) / stride_h;
    const int in_w = (group_out_w - padding_w - output_padding_w) / stride_w;

    for (int out_c = threadIdx.y; out_c < out_channels_per_group; out_c += blockDim.y) {
        scalar_t val = 0;
        for (int k_d = 0; k_d < kernel_d; ++k_d) {
            const int d = in_d + k_d;
            if (d < 0 || d >= input.size(2)) continue;
            for (int k_h = 0; k_h < kernel_h; ++k_h) {
                const int h = in_h + k_h;
                if (h < 0 || h >= input.size(3)) continue;
                for (int k_w = 0; k_w < kernel_w; ++k_w) {
                    const int w = in_w + k_w;
                    if (w < 0 || w >= input.size(4)) continue;
                    for (int in_c = 0; in_c < in_channels_per_group; ++in_c) {
                        val += input[batch_idx][in_c + g*in_channels_per_group][d][h][w] *
                               weight[out_c + g*out_channels_per_group][in_c][k_d][k_h][k_w];
                    }
                }
            }
        }
        output[batch_idx][out_c + g*out_channels_per_group][group_out_d][group_out_h][group_out_w] = val;
    }
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int stride_d,
                                   int stride_h,
                                   int stride_w,
                                   int padding_d,
                                   int padding_h,
                                   int padding_w,
                                   int output_padding_d,
                                   int output_padding_h,
                                   int output_padding_w,
                                   int groups) {
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    // Compute output dimensions
    int out_depth = (in_depth - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    int out_height = (in_height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int out_width = (in_width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, output_options);

    dim3 threads(32, 32);
    dim3 blocks(batch_size, out_depth, out_height);

    // Launch kernel
    conv_transpose3d_kernel<float><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,5,const>(),
        output.packed_accessor<scalar_t,5>(),
        weight.packed_accessor<scalar_t,5,const>(),
        out_channels,
        in_channels,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        output_padding_d,
        output_padding_h,
        output_padding_w,
        groups
    );

    return output;
}
"""

# Compile the custom kernel
conv_transpose3d = load_inline(
    name='conv_transpose3d',
    cpp_sources='',
    cuda_sources=conv_transpose3d_kernel,
    functions=['conv_transpose3d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), 
                 output_padding=(0,0,0), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weights similar to ConvTranspose3d
        kernel_dim = (in_channels, out_channels // groups) + kernel_size
        self.weight = nn.Parameter(torch.empty(kernel_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x):
        # Extract parameters
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        op_d, op_h, op_w = self.output_padding
        groups = self.groups

        # Call the custom CUDA kernel
        output = conv_transpose3d.conv_transpose3d_cuda(
            x, self.weight, stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            op_d, op_h, op_w, groups
        )
        
        return output