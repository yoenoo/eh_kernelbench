import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv1d_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename scalar_t>
__global__ void conv_transpose1d_kernel(const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ weight,
                                       scalar_t* __restrict__ output,
                                       int batches,
                                       int in_channels,
                                       int out_channels,
                                       int kernel_size,
                                       int input_length,
                                       int output_length,
                                       int stride,
                                       int padding,
                                       int dilation,
                                       int groups) {

    const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index >= batches * out_channels * output_length) {
        return;
    }

    int group_size = out_channels / groups;
    int group = output_index / (group_size * output_length * batches);
    int batch = (output_index / (group_size * output_length)) % batches;
    int out_channel = (output_index / output_length) % group_size + group * group_size;
    int out_position = output_index % output_length;

    scalar_t sum = 0;
    for (int kw = 0; kw < kernel_size; ++kw) {
        int in_pos = out_position + padding - kw * dilation;
        if (in_pos < 0 || in_pos >= input_length) {
            continue;
        }
        int in_channel = out_channel % in_channels;
        if (in_channel >= in_channels) {
            continue;
        }
        sum += weight[kw * group_size + out_channel] *
               input[batch * in_channels * input_length + in_channel * input_length + in_pos];
    }
    output[output_index] = sum;
}

torch::Tensor conv_transpose1d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int stride,
                                   int padding,
                                   int output_padding,
                                   int groups) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int kernel_size = weight.size(0);
    const int out_channels = weight.size(1);
    const int output_length = (input_length - 1) * stride + kernel_size - 2 * padding - output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    const int threads = 256;
    const int elements = batch_size * out_channels * output_length;
    const int blocks = (elements + threads - 1) / threads;

    conv_transpose1d_kernel<scalar_t>
        <<<blocks, threads, 0, c10::cuda::get_current_stream()>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            input_length,
            output_length,
            stride,
            padding,
            1, // dilation
            groups);

    return output;
}
"""

conv1d_transpose_cpp_source = """
torch::Tensor conv_transpose1d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int stride,
                                   int padding,
                                   int output_padding,
                                   int groups);
"""

conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=[conv1d_transpose_cpp_source],
    cuda_sources=[conv1d_transpose_source],
    functions=["conv_transpose1d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(kernel_size, out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = conv_transpose1d.conv_transpose1d_cuda(
            x.cuda(),
            self.weight.cuda(),
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )
        if self.bias is not None:
            out += self.bias.view(1, -1, 1)
        return out