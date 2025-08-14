import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1x1 pointwise convolution
pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void pointwise_conv_forward_kernel(const scalar_t* __restrict__ input,
                                             const scalar_t* __restrict__ weight,
                                             scalar_t* __restrict__ output,
                                             const int batch_size,
                                             const int in_channels,
                                             const int out_channels,
                                             const int spatial_size) {

    const int batch_idx = blockIdx.x;
    const int out_channel = threadIdx.x;
    const int spatial_idx = threadIdx.y;

    if (out_channel < out_channels && spatial_idx < spatial_size) {
        scalar_t sum = 0;
        for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
            sum += input[batch_idx * in_channels * spatial_size + in_channel * spatial_size + spatial_idx] *
                   weight[in_channel * out_channels + out_channel];
        }
        output[batch_idx * out_channels * spatial_size + out_channel * spatial_size + spatial_idx] = sum;
    }
}

torch::Tensor pointwise_conv_forward_cuda(torch::Tensor input, torch::Tensor weight) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0); // since weight is [in_channels, out_channels]
    const auto height = input.size(2);
    const auto width = input.size(3);
    const auto spatial_size = height * width;

    auto output = torch::empty({batch_size, out_channels, height, width}, input.options());

    dim3 threads(out_channels, spatial_size);
    dim3 blocks(batch_size);

    const int shared_mem_size = 0;

    // Launch the kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pointwise_conv_forward_cuda", ([&] {
        pointwise_conv_forward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            spatial_size);
    }));

    return output;
}
"""

cpp_source = "torch::Tensor pointwise_conv_forward_cuda(torch::Tensor input, torch::Tensor weight);"

# Load the CUDA kernel
pointwise_conv = load_inline(
    name="pointwise_conv",
    cpp_sources=cpp_source,
    cuda_sources=pointwise_conv_source,
    functions=["pointwise_conv_forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty((in_channels, out_channels)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        # Initialize weights and bias similarly to PyTorch's Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self.bias_ = bias  # To keep track if bias is present
        self.pointwise_conv = pointwise

    def forward(self, x):
        output = pointwise_conv.pointwise_conv_forward_cuda(x, self.weight)
        if self.bias_:
            output += self.bias.view(1, -1, 1, 1)
        return output