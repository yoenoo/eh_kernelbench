import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input,
                                       const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> weight,
                                       torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output,
                                       const int batch_size, const int in_channels, const int out_channels_per_group,
                                       const int input_height, const int input_width,
                                       const int kernel_h, const int kernel_w,
                                       const int stride_h, const int stride_w,
                                       const int padding_h, const int padding_w,
                                       const int dilation_h, const int dilation_w,
                                       const int groups) {

    const int output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1;
    const int output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1;
    
    const int batch_idx = blockIdx.x;
    const int out_channel = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_y = blockIdx.z * blockDim.z + threadIdx.z;
    const int out_x = blockIdx.w * blockDim.w + threadIdx.w;

    if (out_channel >= out_channels_per_group * groups || out_y >= output_height || out_x >= output_width) {
        return;
    }

    const int group = out_channel / out_channels_per_group;
    const int group_out_channel = out_channel % out_channels_per_group;
    const int in_channel_start = group * (in_channels / groups);

    scalar_t val = 0;
    for (int kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
        for (int kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
            const int input_y = (out_y + padding_h - dilation_h * kernel_y) / stride_h;
            const int input_x = (out_x + padding_w - dilation_w * kernel_x) / stride_w;

            if ((out_y + padding_h - dilation_h * kernel_y) % stride_h == 0 &&
                (out_x + padding_w - dilation_w * kernel_x) % stride_w == 0 &&
                input_y >= 0 && input_y < input_height &&
                input_x >= 0 && input_x < input_width) {

                for (int in_channel = in_channel_start; in_channel < in_channel_start + (in_channels / groups); ++in_channel) {
                    val += input[batch_idx][in_channel][input_y][input_x] * 
                           weight[group_out_channel][in_channel - in_channel_start][kernel_y][kernel_x];
                }
            }
        }
    }

    output[batch_idx][out_channel][out_y][out_x] = val;
}

std::tuple<torch::Tensor, torch::Tensor> conv_transpose2d_forward(torch::Tensor input, torch::Tensor weight,
                                                                  int stride_h, int stride_w,
                                                                  int padding_h, int padding_w,
                                                                  int dilation_h, int dilation_w,
                                                                  int groups) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0) * groups;
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1;
    const auto output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 threads(16, 16); // Adjust based on architecture
    dim3 blocks(batch_size, (out_channels + threads.y - 1) / threads.y,
               (output_height + threads.z - 1) / threads.z,
               (output_width + threads.w - 1) / threads.w);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_forward", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            batch_size, in_channels, weight.size(0),
            input_height, input_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups);
    }));

    cudaDeviceSynchronize();
    return std::make_tuple(output, weight); // Return weight for backward (not implemented here)
}
"""

conv_transpose_cpp_source = """
std::tuple<torch::Tensor, torch::Tensor> conv_transpose2d_forward(torch::Tensor input, torch::Tensor weight,
                                                                  int stride_h, int stride_w,
                                                                  int padding_h, int padding_w,
                                                                  int dilation_h, int dilation_w,
                                                                  int groups);
"""

# Compile the CUDA kernel
conv_transpose_module = load_inline(
    name="conv_transpose",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=[
        "conv_transpose2d_forward"
    ],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), 
                 dilation=(1, 1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.randn(out_channels // groups, in_channels // groups, *kernel_size))
        # Bias is omitted for simplicity, but can be added if required
        # self.bias = None
        # if bias:
        #     self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        output = conv_transpose_module.conv_transpose2d_forward(
            x, self.weight, self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self dilation[1],
            self.groups
        )[0]
        return output