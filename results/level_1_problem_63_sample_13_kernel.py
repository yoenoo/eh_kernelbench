import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for 2D convolution
conv2d_cuda_src = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv2d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
                              const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
                              torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
                              int kernel_h, int kernel_w,
                              int stride, int padding,
                              int dilation) {
    int channels_out = weight.size(0);
    int channels_in = weight.size(1);
    int height_in = input.size(2);
    int width_in = input.size(3);
    int height_out = output.size(2);
    int width_out = output.size(3);

    const int H = blockIdx.x * blockDim.x + threadIdx.x;
    const int W = blockIdx.y * blockDim.y + threadIdx.y;
    const int CO = blockIdx.z;

    if (CO >= channels_out || H >= height_out || W >= width_out) return;

    scalar_t sum = 0;
    int h_in = H * stride - padding;
    int w_in = W * stride - padding;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h = h_in + kh * dilation;
            int w = w_in + kw * dilation;
            if (h >= 0 && h < height_in && w >= 0 && w < width_in) {
                for (int CI = 0; CI < channels_in; ++CI) {
                    sum += weight[CO][CI][kh][kw] * input[0][CI][h][w]; // Assuming batch size 1
                }
            }
        }
    }
    output[0][CO][H][W] = sum;
}

torch::Tensor conv2d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation) {
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int batch_size = input.size(0);
    const int channels_in = input.size(1);
    const int height_in = input.size(2);
    const int width_in = input.size(3);
    const int channels_out = weight.size(0);

    int height_out = (height_in + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int width_out = (width_in + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({batch_size, channels_out, height_out, width_out}, output_options);

    dim3 threads(16, 16, 1); // Thread block dimensions
    dim3 blocks((height_out + threads.x - 1)/threads.x, (width_out + threads.y - 1)/threads.y, channels_out);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_forward_cuda", ([&] {
        conv2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            kernel_h, kernel_w, stride, padding, dilation
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv2d_cuda_hdr = """
#include <torch/extension.h>

torch::Tensor conv2d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);
"""

# Compile the inline CUDA code
conv2d_cuda = load_inline(
    name='conv2d_cuda',
    cuda_sources=conv2d_src,
    additionalavenport="",
    extra_cuda_cflags=['-arch=sm_75'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We are reimplementing only the forward pass, so parameters are stored as buffers
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        output = conv2d_cuda.conv2d_forward_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output