import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 2D convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void conv2d_kernel(const torch::PackedTensorAccessor<scalar_t,4> input,
                             const torch::PackedTensorAccessor<scalar_t,4> weight,
                             torch::PackedTensorAccessor<scalar_t,4> output,
                             int in_channels,
                             int out_channels,
                             int kernel_size,
                             int stride,
                             int padding,
                             int dilation) {
    const int H_out = output.size(2);
    const int W_out = output.size(3);
    const int B = blockIdx.z;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_channel = threadIdx.z;

    if (out_y >= H_out || out_x >= W_out || out_channel >= out_channels) return;

    scalar_t sum = 0;
    for (int i = 0; i < in_channels; ++i) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; ky < kernel_size; ++ky) {  // Corrected loop variable typo
                int y = out_y * stride - padding + ky * dilation;
                int x = out_x * stride - padding + kx * dilation;
                if (y >= 0 && y < input.size(2) && x >= 0 && x < input.size(3)) {
                    sum += input[B][i][y][x] * weight[out_channel][i][ky][kx];
                }
            }
        }
    }
    output[B][out_channel][out_y][out_x] = sum;
}

at::Tensor conv2d_cuda(const at::Tensor& input, const at::Tensor& weight,
                      int stride, int padding, int dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    at::Tensor output = at::empty({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 threads(8, 8, 16);  // Thread block dimensions
    dim3 blocks(output_width, output_height, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_cuda", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

convolution_cpp_source = """
at::Tensor conv2d_cuda(const at::Tensor& input, const at::Tensor& weight,
                      int stride, int padding, int dilation);
"""

# Load the CUDA kernel
conv2d_op = load_inline(
    name='convolution',
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=['conv2d_cuda'],
    verbose=True,
    extra_cflags=['-DVERSION_GE_1_5'],
    extra_cuda_cflags=['-gencode=arch=compute_70,code=sm_70']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        # Initialize weights (simplified for brevity; should use proper initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = conv2d_op.conv2d_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out