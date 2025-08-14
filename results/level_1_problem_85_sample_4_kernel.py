import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for depthwise convolution
depthwise_conv_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                           \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void depthwise_conv2d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    const int batch_size, const int channels,
    const int input_height, const int input_width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int dilation_h, const int dilation_w) {
  
    CUDA_1D_KERNEL_LOOP(index, batch_size * channels * output_height * output_width) {
        int w_out = index % output_width;
        int h_out = (index / output_width) % output_height;
        int channel = (index / (output_width * output_height)) % channels;
        int n = index / (channels * output_height * output_width);

        scalar_t val = 0;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride_h - padding_h + kh * dilation_h;
                int w_in = w_out * stride_w - padding_w + kw * dilation_w;
                // Skip input outside padding
                if (h_in >= 0 && h_in < input_height && w_in >=0 && w_in < input_width) {
                    val += input[n][channel][h_in][w_in] *
                           weight[channel][0][kh][kw];
                }
            }
        }
        output[n][channel][h_out][w_out] = val;
    }
}

at::Tensor depthwise_conv2d_forward(
    at::Tensor input,
    at::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w) {
    
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    // Compute output dimensions
    const int output_height = (input_height + 2 * padding_h -
                             dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int output_width = (input_width + 2 * padding_w -
                             dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = at::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int elements = batch_size * channels * output_height * output_width;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_forward", ([&] {
        depthwise_conv2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, channels, input_height, input_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

depthwise_conv_forward_source = (
    "at::Tensor depthwise_conv2d_forward(at::Tensor input, at::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);"
)

# Compile the inline CUDA code
depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_forward_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_forward"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0,
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size_h, kernel_size_w)
        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)
        self.dilation = (dilation_h, dilation_w)
        self.groups = groups
        self.bias = bias

        # Initialize convolution weights manually
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size_h, kernel_size_w))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        self.depthwise_conv = depthwise_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.depthwise_conv.depthwise_conv2d_forward(
            x.cuda(), 
            self.weight.cuda(), 
            self.stride[0], 
            self.stride[1], 
            self.padding[0], 
            self.padding[1],
            self.dilation[0],
            self.dilation[1]
        )
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output