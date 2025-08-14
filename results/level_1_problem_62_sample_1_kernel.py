import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for convolution with asymmetric kernels
conv2d_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define BLOCK_SIZE 32

namespace {

template <typename scalar_t>
__global__ void asymmetric_convolution(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_h, int kernel_w,
    int stride, int padding_h, int padding_w,
    int dilation_h, int dilation_w) {

    int output_h = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int output_w = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    const int b = blockIdx.z;
    const int out_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int out_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (out_y < output_h && out_x < output_w) {
        for (int c_out = threadIdx.z; c_out < out_channels; c_out += blockDim.z) {
            scalar_t sum = 0;
            for (int k_h = 0; k_h < kernel_h; ++k_h) {
                int input_y = out_y * stride - padding_h + k_h * dilation_h;
                if (input_y < 0 || input_y >= input_height) continue;
                for (int k_w = 0; k_w < kernel_w; ++k_w) {
                    int input_x = out_x * stride - padding_w + k_w * dilation_w;
                    if (input_x < 0 || input_x >= input_width) continue;
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        sum += input[b][c_in][input_y][input_x] *
                               weight[c_out][c_in][k_h][k_w];
                    }
                }
            }
            output[b][c_out][out_y][out_x] = sum;
        }
    }
}

} // anonymous namespace

torch::Tensor asymmetric_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels = weight.size(0);

    int output_h = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int output_w = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocks(
        (output_w + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (output_h + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "asymmetric_convolution", ([&] {
        asymmetric_convolution<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_h, kernel_w,
            stride, padding_h, padding_w,
            dilation_h, dilation_w);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor asymmetric_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w);
"""

# Compile the custom convolution kernel
asymmetric_conv2d = load_inline(
    name='asymmetric_conv2d',
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_kernel,
    functions=['asymmetric_conv2d_cuda'],
    verbose=True,
    extra_cflags=['-D_WITH_CUDA'],
    extra_cuda_cflags=['-lineinfo']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(padding, int):
            self.padding_h, self.padding_w = padding, padding
        else:
            self.padding_h, self.padding_w = padding
        self.dilation_h, self.dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights
        weight_shape = (out_channels, in_channels // groups, kernel_size[0], kernel_size[1])
        self.weight = nn.Parameter(torch.empty(weight_shape).normal_(0, 0.02))
        
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels).normal_(0, 0.02))
        else:
            self.bias_param = None

        # Initialize custom op
        self.convolver = asymmetric_conv2d

    def forward(self, x):
        output = self.convolver.asymmetric_conv2d_cuda(
            x, self.weight, 
            self.stride, 
            self.padding_h, self.padding_w,
            self.dilation_h, self.dilation_w
        )

        if self.bias_param is not None:
            output += self.bias_param.view(1, -1, 1, 1)

        return output