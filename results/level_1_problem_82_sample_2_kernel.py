import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for depthwise convolution
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int kernel_size, int stride) {

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int output_H = output.size(2);
    int output_W = output.size(3);

    // Calculate output indices
    int n = blockIdx.x;
    int c = blockIdx.y;
    int oh = blockIdx.z;
    int ow = threadIdx.x;

    scalar_t sum = 0;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_in = oh * stride + kh;
            int w_in = ow * stride + kw;
            if (h_in < H && w_in < W) {
                sum += input[n][c][h_in][w_in] * weight[c][0][kh][kw];
            }
        }
    }

    output[n][c][oh][ow] = sum;
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride) {
    auto output_height = (input.size(2) - kernel_size) / stride + 1;
    auto output_width = (input.size(3) - kernel_size) / stride + 1;
    auto output = torch::zeros({input.size(0), input.size(1), output_height, output_width}, input.options());

    dim3 blocks(input.size(0), input.size(1), output_height);
    dim3 threads(output_width);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            kernel_size, stride);
    }));

    return output;
}
"""

depthwise_conv_cpp_source = (
    "torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride);"
)

# Compile the inline CUDA code for depthwise convolution
depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Initialize weights like PyTorch's Conv2d
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # He initialization

        if bias:
            self.bias_param = nn.Parameter(torch.empty(in_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)
        else:
            self.bias_param = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = depthwise_conv.depthwise_conv2d_cuda(x, self.weight, self.kernel_size, self.stride)
        if self.bias_param is not None:
            out = out + self.bias_param.view(1, -1, 1, 1)
        return out