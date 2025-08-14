import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for pointwise convolution (equivalent to 1x1 Conv2D)
pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void pointwise_conv_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int batch_size, int in_channels, int out_channels, int height, int width) {

    const int HW = height * width;
    const int BC = blockIdx.x * blockDim.x + threadIdx.x;

    if (BC < batch_size * out_channels) {
        const int b = BC / out_channels;
        const int c_out = BC % out_channels;
        const int c_in = 0; // assuming all input channels are summed (if kernel is 1x1)
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                output[b][c_out][h][w] += input[b][c_in][h][w] * weight[c_out][c_in][0][0];
            }
        }
    }
}

torch::Tensor pointwise_conv_forward_cuda(torch::Tensor input, torch::Tensor weight) {
    auto output = torch::zeros({input.size(0), weight.size(0), input.size(2), input.size(3)}, input.options());

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int height = input.size(2);
    const int width = input.size(3);

    const int threads = 256;
    const int blocks = (batch_size * out_channels + threads -1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "pointwise_conv_forward_cuda", ([&]{
        pointwise_conv_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, in_channels, out_channels, height, width
        );
    }));

    return output;
}
"""

pointwise_conv_header = """
torch::Tensor pointwise_conv_forward_cuda(torch::Tensor input, torch::Tensor weight);
"""

pointwise_conv = load_inline(
    name="pointwise_conv",
    cpp_sources=pointwise_conv_header,
    cuda_sources=pointwise_conv_source,
    functions=["pointwise_conv_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, 1, 1)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.pointwise_conv_forward = pointwise_conv

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = self.pointwise_conv_forward.pointwise_conv_forward_cuda(x, self.weight)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output