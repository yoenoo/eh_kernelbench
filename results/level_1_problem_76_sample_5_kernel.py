import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

def _get_conv1d_params(x: torch.Tensor, conv: nn.Conv1d):
    batch_size, in_channels, length = x.size()
    out_channels = conv.out_channels
    kernel_size = conv.kernel_size[0]
    stride = conv.stride[0]
    dilation = conv.dilation[0]
    padding = conv.padding[0]
    return batch_size, in_channels, out_channels, length, kernel_size, stride, dilation, padding

conv1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename scalar_t>
__global__ void custom_conv1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int length,
    const int kernel_size,
    const int stride,
    const int dilation,
    const int padding
) {
    int batch = blockIdx.x;
    int out_channel = blockIdx.y;
    int out_pos = threadIdx.x;

    int output_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    if (out_pos >= output_length) return;

    scalar_t sum = bias[out_channel];
    int in_pos = -padding + out_pos * stride;
    for (int i = 0; i < kernel_size; ++i) {
        if (in_pos >= 0 && in_pos < length) {
            for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
                sum += weight[out_channel * in_channels * kernel_size + in_channel * kernel_size + i] *
                    input[batch * in_channels * length + in_channel * length + in_pos];
            }
        }
        in_pos += dilation;
    }
    output[batch * out_channels * output_length + out_channel * output_length + out_pos] = sum;
}

torch::Tensor custom_conv1d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int dilation,
    int padding
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int length = input.size(2);
    const int kernel_size = weight.size(2);

    int output_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    dim3 blocks(batch_size, out_channels);
    dim3 threads(output_length);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv1d_kernel", ([&] {
        custom_conv1d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            length,
            kernel_size,
            stride,
            dilation,
            padding
        );
    }));

    return output;
}
"""

conv1d_cpp = """
torch::Tensor custom_conv1d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int dilation,
    int padding
);
"""

conv1d_cuda = load_inline(
    name="custom_conv1d",
    cpp_sources=conv1d_cpp,
    cuda_sources=conv1d_source,
    functions=["custom_conv1d"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias_weight = nn.Parameter(torch.empty(out_channels)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_weight is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, length = x.size()
        padding = self.conv.padding[0]  # Get padding from original conv parameters
        return conv1d_cuda.custom_conv1d(
            x.cuda(),
            self.weight,
            self.bias_weight if self.bias else torch.zeros(1).cuda(),
            self.stride,
            self.dilation,
            padding
        )