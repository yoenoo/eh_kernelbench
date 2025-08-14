import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from torch.nn.modules.utils import _triple, _single

# Define the custom CUDA kernel for 3D ConvTranspose
conv_transpose3d_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

using at::cuda::CUDAGuard;

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

namespace {

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(const scalar_t* __restrict__ input,
                                      const scalar_t* __restrict__ weight,
                                      scalar_t* __restrict__ output,
                                      int64_t batch, int64_t in_channels,
                                      int64_t depth, int64_t height, int64_t width,
                                      int64_t out_depth, int64_t out_height, int64_t out_width,
                                      int64_t kernel_size_depth, int64_t kernel_size_height, int64_t kernel_size_width,
                                      int64_t stride_depth, int64_t stride_height, int64_t stride_width,
                                      int64_t padding_depth, int64_t padding_height, int64_t padding_width,
                                      int64_t output_padding_depth, int64_t output_padding_height, int64_t output_padding_width,
                                      int64_t dilation_depth, int64_t dilation_height, int64_t dilation_width,
                                      int64_t out_channels, int64_t groups) {
    // This kernel implements the transpose convolution computation
    const int64_t output_depth_stride = out_channels * out_depth * out_height * out_width;
    const int64_t output_channel_stride = out_depth * out_height * out_width;
    const int64_t output_depth_stride_per_batch = out_channels * out_depth * out_height * out_width / batch;

    CUDA_1D_KERNEL_LOOP(index, batch * out_channels * out_depth * out_height * out_width) {
        int64_t batch_id = index / output_depth_stride;
        int64_t out_ch = (index % output_depth_stride) / output_channel_stride;
        int64_t d_out = (index % output_channel_stride) / (out_height * out_width);
        int64_t h_out = (index % (out_height * out_width)) / out_width;
        int64_t w_out = index % out_width;

        // Compute the input coordinates
        int64_t d_in = (d_out - padding_depth - output_padding_depth) / stride_depth;
        int64_t h_in = (h_out - padding_height - output_padding_height) / stride_height;
        int64_t w_in = (w_out - padding_width - output_padding_width) / stride_width;

        scalar_t val = 0;
        // Iterate over kernel elements
        for (int64_t kd = 0; kd < kernel_size_depth; ++kd) {
            int64_t delta_d = d_in * stride_depth + padding_depth + output_padding_depth + kd * dilation_depth - padding_depth;
            if (delta_d < 0 || delta_d >= depth) continue;
            int64_t d_kernel = kd;

            for (int64_t kh = 0; kh < kernel_size_height; ++kh) {
                int64_t delta_h = h_out - kh * dilation_height;
                if (delta_h < padding_height || delta_h >= padding_height + height * stride_height) continue;
                int64_t h_kernel = kh;

                for (int64_t kw = 0; kw < kernel_size_width; ++kw) {
                    int64_t delta_w = w_out - kw * dilation_width;
                    if (delta_w < padding_width || delta_w >= padding_width + width * stride_width) continue;
                    int64_t w_kernel = kw;

                    const int64_t weight_offset = (out_ch * in_channels / groups) * kernel_size_depth * kernel_size_height * kernel_size_width +
                                                (d_kernel * kernel_size_height + h_kernel) * kernel_size_width + w_kernel;
                    const int64_t input_offset = batch_id * in_channels * depth * height * width +
                                                ((out_ch / (in_channels / groups)) * in_channels / groups) * depth * height * width +
                                                (delta_d * height + (delta_h - padding_height)/stride_height) * width +
                                                (delta_w - padding_width)/stride_width;

                    val += weight[weight_offset] * input[input_offset];
                }
            }
        }
        output[index] = val;
    }
}

} // end namespace

std::vector<int64_t> output_shape(int64_t in_depth, int64_t in_height, int64_t in_width, int64_t kernel_depth, int64_t kernel_height, int64_t kernel_width,
                                  int64_t stride_depth, int64_t stride_height, int64_t stride_width,
                                  int64_t padding_depth, int64_t padding_height, int64_t padding_width,
                                  int64_t output_padding_depth, int64_t output_padding_height, int64_t output_padding_width,
                                  int64_t dilation_depth, int64_t dilation_height, int64_t dilation_width) {
    int64_t out_depth = (in_depth - 1) * stride_depth - 2 * padding_depth + dilation_depth * (kernel_depth - 1) + output_padding_depth + 1;
    int64_t out_height = (in_height - 1) * stride_height - 2 * padding_height + dilation_height * (kernel_height - 1) + output_padding_height + 1;
    int64_t out_width = (in_width - 1) * stride_width - 2 * padding_width + dilation_width * (kernel_width - 1) + output_padding_width + 1;
    return {out_depth, out_height, out_width};
}

at::Tensor conv_transpose3d_cuda(const at::Tensor& input,
                                 const at::Tensor& weight,
                                 int64_t stride_depth, int64_t stride_height, int64_t stride_width,
                                 int64_t padding_depth, int64_t padding_height, int64_t padding_width,
                                 int64_t output_padding_depth, int64_t output_padding_height, int64_t output_padding_width,
                                 int64_t dilation_depth, int60_t dilation_height, int64_t dilation_width,
                                 int64_t groups) {
    AT_ASSERTM(input.dim() == 5, "Expected 5D input");
    AT_ASSERTM(weight.dim() == 5, "Expected 5D weight");

    const int64_t batch = input.size(0);
    const int64_t in_channels = input.size(1);
    const int64_t kernel_depth = weight.size(2);
    const int64_t kernel_height = weight.size(3);
    const int64_t kernel_width = weight.size(4);
    const int64_t out_channels = weight.size(0);

    auto input_size = input.sizes();
    int64_t in_depth = input_size[2], in_height = input_size[3], in_width = input_size[4];

    auto out_dims = output_shape(in_depth, in_height, in_width,
                                kernel_depth, kernel_height, kernel_width,
                                stride_depth, stride_height, stride_width,
                                padding_depth, padding_height, padding_width,
                                output_padding_depth, output_padding_height, output_padding_width,
                                dilation_depth, dilation_height, dilation_width);
    int64_t out_depth = out_dims[0], out_height = out_dims[1], out_width = out_dims[2];

    auto output = at::empty({batch, out_channels, out_depth, out_height, out_width}, input.type());
    
    const int threads = 1024;
    int64_t total_elements = batch * out_channels * out_depth * out_height * out_width;
    int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_cuda", ([&] {
        conv_transpose3d_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch, in_channels, in_depth, in_height, in_width,
            out_depth, out_height, out_width,
            kernel_depth, kernel_height, kernel_width,
            stride_depth, stride_height, stride_width,
            padding_depth, padding_height, padding_width,
            output_padding_depth, output_padding_height, output_padding_width,
            dilation_depth, dilation_height, dilation_width,
            out_channels, groups);
    }));

    return output;
}

"""

conv_transpose3d_cpp_source = """
#include <torch/extension.h>

at::Tensor conv_transpose3d_cuda(const at::Tensor& input,
                                 const at::Tensor& weight,
                                 int64_t stride_depth, int64_t stride_height, int64_t stride_width,
                                 int64_t padding_depth, int64_t padding_height, int64_t padding_width,
                                 int64_t output_padding_depth, int64_t output_padding_height, int64_t output_padding_width,
                                 int64_t dilation_depth, int64_t dilation_height, int64_t dilation_width,
                                 int64_t groups);
"""

# Compile the custom CUDA kernel
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.output_padding = _triple(output_padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.bias = bias

        # Initialize weight like PyTorch ConvTranspose3d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return conv_transpose3d.conv_transpose3d_cuda(
            x, self.weight,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2],
            self.dilation[0], self.dilation[1], self.dilation[2],
            self.groups
        )