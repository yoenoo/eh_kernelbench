import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 2D convolution
convolution_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv2d_kernel(const torch::PackedTensorAccessor<scalar_t,4> input,
                             const torch::PackedTensorAccessor<scalar_t,4> weight,
                             torch::PackedTensorAccessor<scalar_t,4> output,
                             int kernel_h, int kernel_w,
                             int stride_h, int stride_w,
                             int padding_h, int padding_w,
                             int dilation_h, int dilation_w,
                             int groups) {
    const int batch_idx = blockIdx.x;
    const int out_channel = blockIdx.y;
    const int out_h = threadIdx.x;
    const int out_w = threadIdx.y;

    scalar_t sum = 0;
    const int in_channels_per_group = input.size(1) / groups;
    const int out_channels_per_group = output.size(1) / groups;
    const int group = out_channel / out_channels_per_group;

    for (int k_h = 0; k_h < kernel_h; ++k_h) {
        for (int k_w = 0; k_w < kernel_w; ++k_w) {
            const int h_in = -padding_h + out_h * stride_h + k_h * dilation_h;
            const int w_in = -padding_w + out_w * stride_w + k_w * dilation_w;

            if (h_in >= 0 && h_in < input.size(2) && w_in >= 0 && w_in < input.size(3)) {
                for (int i_c = 0; i_c < in_channels_per_group; ++i_c) {
                    sum += input[batch_idx][group * in_channels_per_group + i_c][h_in][w_in] * 
                           weight[out_channel][i_c][k_h][k_w];
                }
            }
        }
    }

    output[batch_idx][out_channel][out_h][out_w] = sum;
}

std::vector<torch::Tensor> custom_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups) {
    
    const auto batch_size = input.size(0);
    const auto out_channels = weight.size(0);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);
    
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);
    
    const auto out_height = (input_height + 2 * padding_h - 
                        dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const auto out_width = (input_width + 2 * padding_w - 
                       dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    dim3 blocks(batch_size, out_channels);
    dim3 threads(out_height, out_width);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_forward", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups);
    }));
    
    cudaDeviceSynchronize();
    return {output};
}
"""

convolution_cpp_source = """
std::vector<torch::Tensor> custom_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups);
"""

# Compile the custom CUDA kernel
custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["custom_conv2d"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
            
    def forward(self, x):
        outputs = custom_conv.custom_conv2d(
            x, 
            self.weight, 
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self dilation[1],
            self.groups
        )[0]
        
        if self.bias is not None:
            outputs += self.bias.view(1, -1, 1, 1)
            
        return outputs