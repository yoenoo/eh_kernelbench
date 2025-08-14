import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
conv2d_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.h>
#include <ATen/cuda/CUDASubstensorImpl.cu>
#include <ATen/cuda/CUDAGeneratorImpl.cu>

template <typename scalar_t>
__global__ void conv2d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
                                     const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
                                     torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
                                     int kernel_h, int kernel_w,
                                     int stride, int padding_h, int padding_w,
                                     int dilation_h, int dilation_w,
                                     int groups) {

    const int n = blockIdx.x;
    const int c_out = blockIdx.y;
    const int y1 = threadIdx.y;
    const int x1 = threadIdx.x;

    const int batch_size = output.size(0);
    const int channels_out = output.size(1);
    const int height_out = output.size(2);
    const int width_out = output.size(3);

    const int channels_in = input.size(1);
    const int height_in = input.size(2);
    const int width_in = input.size(3);

    scalar_t sum = 0;
    for (int i = 0; i < groups; i++) {
        const int c_in = (c_out * groups + i) / groups;
        for (int ky = 0; ky < kernel_h; ky++) {
            for (int kx = 0; kx < kernel_w; kx++) {
                const int h = -padding_h + y1 * stride + ky * dilation_h;
                const int w = -padding_w + x1 * stride + kx * dilation_w;
                if (h >= 0 && h < height_in && w >= 0 && w < width_in) {
                    sum += input[n][c_in][h][w] * weight[c_out][c_in][ky][kx];
                }
            }
        }
    }
    output[n][c_out][y1][x1] = sum;
}

at::Tensor conv2d_forward_cuda(at::Tensor input, at::Tensor weight, int stride, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups) {
    const int batch_size = input.size(0);
    const int channels_in = input.size(1);
    const int height_in = input.size(2);
    const int width_in = input.size(3);

    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int channels_out = weight.size(0);
    const int height_out = (height_in + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    const int width_out = (width_in + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = at::empty({batch_size, channels_out, height_out, width_out}, input.options());

    dim3 threads(kernel_w, kernel_h);
    dim3 blocks(batch_size, channels_out);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_forward_cuda", ([&] {
        conv2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            kernel_h, kernel_w,
            stride, padding_h, padding_w,
            dilation_h, dilation_w,
            groups);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = "at::Tensor conv2d_forward_cuda(at::Tensor input, at::Tensor weight, int stride, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups);"

# Compile the inline CUDA code
conv2d_custom = load_inline(
    name="conv2d_custom",
    cpp_sources=cpp_source,
    cuda_sources=conv2d_kernel,
    functions=["conv2d_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.bias = bias

        # Initialize convolution weights
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        self.conv2d_custom = conv2d_custom

    def forward(self, x):
        output = self.conv2d_custom.conv2d_forward_cuda(
            x, self.weight, self.stride, self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1], self.groups)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output