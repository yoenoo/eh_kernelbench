import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for depthwise convolution
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int batch_size, int in_channels, int input_height, int input_width,
    int kernel_size, int stride, int padding) {

    int N = blockIdx.x;
    int C = blockIdx.y;
    int H_out = blockIdx.z;
    int W_out = threadIdx.x;

    int H_start = H_out * stride - padding;
    int W_start = W_out * stride - padding;

    scalar_t sum = 0;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h = H_start + kh;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int w = W_start + kw;
            if (h >= 0 && h < input_height && w >= 0 && w < input_width) {
                sum += input[N][C][h][w] * weight[C][0][kh][kw];
            }
        }
    }
    output[N][C][H_out][W_out] = sum;
}

std::tuple<torch::Tensor> depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size,
    int stride,
    int padding) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    const auto output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, in_channels, output_height, output_width}, input.options());

    dim3 threads_per_block(256);
    dim3 num_blocks(
        batch_size,
        in_channels,
        output_height
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, in_channels, input_height, input_width,
            kernel_size, stride, padding);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

depthwise_conv_cpp_source = """
std::tuple<torch::Tensor> depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size,
    int stride,
    int padding);
"""

depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x):
        output = depthwise_conv.depthwise_conv2d_cuda(
            x, self.weight, self.kernel_size, self.stride, self.padding)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output
"""

# Test code (not part of the optimized model)
if __name__ == "__main__":
    batch_size = 16
    in_channels = 64
    kernel_size = 3
    width = 512
    height = 512
    stride = 1
    padding = 0

    model = ModelNew(in_channels, kernel_size, stride, padding, bias=True)
    x = torch.rand(batch_size, in_channels, height, width).cuda()
    y = model(x)
    print(y.shape)