import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

__global__ void pointwise_conv_forward(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    bool has_bias) {

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * out_channels * height * width)
        return;

    int w = out_idx % width;
    int h = (out_idx / width) % height;
    int oc = (out_idx / (height * width)) % out_channels;
    int n = out_idx / (out_channels * height * width);

    float sum = 0;
    for (int ic = 0; ic < in_channels; ++ic) {
        int in_offset = n * in_channels * height * width +
                        ic * height * width +
                        h * width + w;
        int weight_offset = oc * in_channels + ic;
        sum += input[in_offset] * weight[weight_offset];
    }

    if (has_bias) {
        sum += bias[oc];
    }

    int out_offset = n * out_channels * height * width +
                     oc * height * width +
                     h * width + w;
    output[out_offset] = sum;
}

torch::Tensor pointwise_conv_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    bool has_bias) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0); // since weight is [out_channels, in_channels]
    const auto height = input.size(2);
    const auto width = input.size(3);

    auto output = torch::empty({batch_size, out_channels, height, width}, input.options());

    dim3 threads(256);
    dim3 blocks((batch_size * out_channels * height * width + threads.x - 1) / threads.x);

   点wise_conv_forward<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width,
        has_bias
    );

    cudaDeviceSynchronize();
    return output;
}
"""

pointwise_conv_cpp_src = """
torch::Tensor pointwise_conv_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    bool has_bias);
"""

pointwise_conv_module = load_inline(
    name="pointwise_conv",
    cpp_sources=pointwise_conv_cpp_src,
    cuda_sources=pointwise_conv_source,
    functions=["pointwise_conv_forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias similar to nn.Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.has_bias = bias
        self.forward_op = pointwise_conv_module.pointwise_conv_forward_cuda

    def reset_parameters(self):
        # Initialize weights similar to PyTorch's Conv2d initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_op(x, self.weight, self.bias if self.has_bias else torch.empty(0), self.has_bias)