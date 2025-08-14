import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized Conv2d
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void optimized_conv2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int stride, int padding, int dilation, int groups) {

    const int B = blockIdx.z;
    const int out_channel = blockIdx.y;
    const int out_h = blockIdx.x;
    const int out_w = threadIdx.x;

    scalar_t sum = 0;
    for (int group = 0; group < groups; ++group) {
        for (int k = 0; k < weight.size(2); ++k) {
            for (int l = 0; l < weight.size(3); ++l) {
                int in_h = out_h * stride + k * dilation - padding;
                int in_w = out_w * stride + l * dilation - padding;
                if (in_h >= 0 && in_h < input.size(2) && in_w >= 0 && in_w < input.size(3)) {
                    for (int c = 0; c < weight.size(1); ++c) {
                        sum += weight[out_channel][c][k][l] * 
                               input[B][group * weight.size(1) + c][in_h][in_w];
                    }
                }
            }
        }
    }
    output[B][out_channel][out_h][out_w] = sum;
}

torch::Tensor optimized_conv2d(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation, int groups) {
    const auto batch_size = input.size(0);
    const auto out_channels = weight.size(0);
    const auto in_channels_per_group = weight.size(1);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    auto output_height = (input.size(2) + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    auto output_width = (input.size(3) + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output_options = torch::TensorOptions().like(input);
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, output_options);

    const int threads = 1024;
    dim3 blocks(output_width, out_channels, batch_size);
    dim3 threads_per_block(threads);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "optimized_conv2d", ([&] {
        optimized_conv2d_kernel<scalar_t><<<blocks, threads_per_block, 0, c10::cuda::getStream()>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            stride, padding, dilation, groups);
    }));

    return output;
}
"""

convolution_cpp_source = "torch::Tensor optimized_conv2d(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation, int groups);"

convolution_ops = load_inline(
    name="convolution_ops",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["optimized_conv2d"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        outputs = convolution_ops.optimized_conv2d(
            x, 
            self.weight, 
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
        if self.bias is not None:
            outputs += self.bias.view(1, -1, 1, 1)
        return outputs