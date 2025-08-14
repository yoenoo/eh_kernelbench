import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom pointwise convolution CUDA kernel implementation
pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void pointwise_conv_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int B, int IC, int OC, int H, int W) {

    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= B * OC * H * W) return;

    const int w = output_idx % W;
    int remaining = output_idx / W;
    const int h = remaining % H;
    remaining /= H;
    const int oc = remaining % OC;
    const int b = remaining / OC;

    scalar_t sum = 0;
    for (int ic = 0; ic < IC; ic++) {
        sum += input[b][ic][h][w] * weight[0][ic][0][0]; // Kernel size 1x1, so no spatial dimensions
    }
    output[b][oc][h][w] = sum;
}

torch::Tensor pointwise_conv_forward_cuda(torch::Tensor input, torch::Tensor weight) {
    const auto B = input.size(0);
    const auto IC = input.size(1);
    const auto OC = weight.size(0); // weight is [OC, IC, 1, 1]
    const auto H = input.size(2);
    const auto W = input.size(3);

    auto output = torch::zeros({B, OC, H, W}, input.options());

    const int total_elements = B * OC * H * W;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "pointwise_conv_forward_cuda", ([&] {
        pointwise_conv_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            B, IC, OC, H, W);
    }));

    return output;
}
"""

pointwise_conv_cpp_source = "torch::Tensor pointwise_conv_forward_cuda(torch::Tensor input, torch::Tensor weight);"

# Compile the CUDA kernel
pointwise_conv = load_inline(
    name="pointwise_conv",
    cpp_sources=pointwise_conv_cpp_source,
    cuda_sources=pointwise_conv_source,
    functions=["pointwise_conv_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, 1, 1)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        # Initialize weights and bias similar to PyTorch's default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        self.bias_flag = bias
        self.pointwise_conv = pointwise_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.pointwise_conv.pointwise_conv_forward_cuda(x, self.weight)
        if self.bias_flag:
            # Add bias here using element-wise addition on the output
            # Using batch matrix multiplication for efficient bias addition
            bias_view = self.bias.view(1, -1, 1, 1)
            output += bias_view
        return output