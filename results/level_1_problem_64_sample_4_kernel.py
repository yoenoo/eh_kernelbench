import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv1d_transpose_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 1024

template <typename scalar_t>
__global__ void conv_transpose1d_kernel(const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ weight,
                                       scalar_t* __restrict__ output,
                                       int output_size,
                                       int batch_size,
                                       int in_channels,
                                       int out_channels,
                                       int kernel_size,
                                       int stride,
                                       int padding,
                                       int dilation,
                                       int output_padding,
                                       int groups) {

    const int output_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_id >= output_size) return;

    int output_col = output_id % output_size;
    int oc = (output_id / output_size) % out_channels;
    int n = output_id / (out_channels * output_size);

    int group = oc / (out_channels / groups);
    oc = oc % (out_channels / groups);

    int in_group = in_channels / groups;
    scalar_t val = 0;

    for (int k = 0; k < kernel_size; ++k) {
        int input_col = (output_col + padding - k * dilation - output_padding) / stride;
        if ((output_col + padding - k * dilation - output_padding) % stride == 0 &&
            input_col >= 0 && input_col < (output_size / stride)) {
            for (int ic = 0; ic < in_group; ++ic) {
                val += weight[oc * in_group * kernel_size + ic * kernel_size + k] *
                       input[n * in_channels * (output_size/stride) + 
                             (group * in_group + ic) * (output_size/stride) + input_col];
            }
        }
    }

    output[output_id] = val;
}

torch::Tensor conv1d_transpose_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int stride,
                                   int padding,
                                   int output_padding,
                                   int groups) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_length = input.size(2);
    const int out_channels = weight.size(0) * groups;
    const int kernel_size = weight.size(2);
    const int dilation = 1; // Assuming no dilation in transposed conv

    int out_length = (in_length - 1) * stride + kernel_size - 2 * padding + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, out_length}, input.options());

    const int threads = BLOCK_SIZE;
    const int elements = batch_size * out_channels * out_length;
    const int blocks = (elements + threads - 1) / threads;

    const int weight_size = weight.numel();
    auto weight_data = weight.contiguous();

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose1d", ([&] {
        conv_transpose1d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight_data.data<scalar_t>(),
            output.data<scalar_t>(),
            out_length,
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            output_padding,
            groups);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv1d_transpose_cpp_source = """
torch::Tensor conv1d_transpose_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int stride,
                                   int padding,
                                   int output_padding,
                                   int groups);
"""

conv1d_transpose_ops = load_inline(
    name="conv1d_transpose",
    cpp_sources=conv1d_cpp_source,
    cuda_sources=conv1d_transpose_source,
    functions=["conv1d_transpose_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # Initialize weights
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x):
        outputs = conv1d_transpose_ops.conv1d_transpose_cuda(
            x, self.weight, self.stride, self.padding, self.output_padding, self.groups)
        if self.bias is not None:
            outputs += self.bias.view(1, -1, 1)
        return outputs