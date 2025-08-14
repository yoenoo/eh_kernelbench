import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Custom pointwise convolution kernel
at::Tensor pointwise_conv_cuda(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias) {
    const int batch = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0); // Since weight is [out_channels, in_channels]

    auto output = at::empty({batch, out_channels, height, width}, input.options());

    // Launch CUDA kernel
    dim3 threads(256);
    dim3 blocks(batch * height * width + threads.x - 1) / threads.x;

    // Launch the kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "pointwise_conv_cuda", ([&] {
        const auto in_data = input.data<scalar_t>();
        const auto weight_data = weight.data<scalar_t>();
        const auto *bias_data = (bias.defined()) ? bias.data<scalar_t>() : nullptr;
        auto out_data = output.data<scalar_t>();

        pointwise_conv_kernel<<<blocks, threads>>>(
            batch, in_channels, height, width, out_channels,
            in_data, weight_data, bias_data, out_data);
    }));

    return output;
}

template <typename scalar_t>
__global__ void pointwise_conv_kernel(
    const int batch, const int in_channels, const int height, const int width, const int out_channels,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * height * width) return;

    const int b = idx / (height * width);
    int rem = idx % (height * width);
    const int h = rem / width;
    const int w = rem % width;

    for (int oc = 0; oc < out_channels; ++oc) {
        scalar_t sum = 0;
        for (int ic = 0; ic < in_channels; ++ic) {
            sum += input[b * in_channels * height * width + ic * height * width + h * width + w] *
                   weight[oc * in_channels + ic];
        }
        if (bias) sum += bias[oc];
        output[b * out_channels * height * width + oc * height * width + h * width + w] = sum;
    }
}
"""

pointwise_conv_cpp_src = "at::Tensor pointwise_conv_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&);"

# Compile the CUDA code
pointwise_conv = load_inline(
    name='pointwise_conv',
    cpp_sources=pointwise_conv_cpp_src,
    cuda_sources=pointwise_conv_source,
    functions=['pointwise_conv_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.register_buffer('weight', torch.empty((out_channels, in_channels)))
        if bias:
            self.register_buffer('bias', torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize weights and bias like PyTorch's Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return pointwise_conv.pointwise_conv_cuda(x, self.weight, self.bias if self.bias is not None else x.new_zeros(0))

    def __repr__(self):
        return f"ModelNew({self.weight.size(1)}, {self.weight.size(0)}, bias={self.bias is not None})"