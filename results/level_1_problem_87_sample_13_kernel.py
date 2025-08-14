import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for pointwise convolution (equivalent to 1x1 Conv2D)
pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void pointwise_conv_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int batch_size, int in_channels, int out_channels, int height, int width) {

    const int batch = blockIdx.x;
    const int oh = threadIdx.z;
    const int ow = threadIdx.y;
    const int ic = threadIdx.x;

    // Each thread processes a single output element
    const int output_element = ow + width*(oh) + width*height*(ic) + width*height*in_channels*(batch);

    scalar_t sum = 0;
    for (int k = 0; k < in_channels; ++k) {
        sum += input[batch][k][oh][ow] * weight[k][ic];
    }
    output[batch][ic][oh][ow] = sum;
}

torch::Tensor pointwise_conv_forward_cuda(torch::Tensor input, torch::Tensor weight) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    const dim3 threads(in_channels, width, height);
    const dim3 blocks(batch_size);

    const int num_threads = in_channels * width * height;
    if (num_threads > 1024) {
        // Reorganize threads if needed for larger dimensions
        // This is a simplified version and may need adjustment for optimal performance
    }

    AT_DISPATCH_FLOATING_TYPES(input.type(), "pointwise_conv_forward", ([&] {
        pointwise_conv_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, in_channels, out_channels, height, width);
    }));

    return output;
}
"""

pointwise_conv_cpp_source = (
    "torch::Tensor pointwise_conv_forward_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code
pointwise_conv = load_inline(
    name="pointwise_conv",
    cpp_sources=pointwise_conv_cpp_source,
    cuda_sources=pointwise_conv_source,
    functions=["pointwise_conv_forward_cuda"],
    verbose=True,
    extra_cflags=["-g"],
    extra_ldflags=["-lstdc++"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty((in_channels, out_channels)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        # Initialize weights and bias (matches PyTorch's default initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        self.pointwise_conv = pointwise_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.pointwise_conv.pointwise_conv_forward_cuda(x, self.weight)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output