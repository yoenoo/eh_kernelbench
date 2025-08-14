import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose1d
conv_transpose_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename scalar_t>
__global__ void conv_transpose1d_backward_data_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    const int stride, const int padding, const int output_padding,
    const int kernel_size, const int groups) {

    const int batch = blockIdx.x;
    const int in_channel_group = blockIdx.y;
    const int group = blockIdx.z;
    const int out_channel = threadIdx.x;

    const int in_channel = group * (weight.size(0)/groups) + in_channel_group;
    const int out_channel_total = group * (weight.size(1)/groups) + out_channel;

    for (int i = 0; i < input.size(2); ++i) {
        const int output_index = i * stride - padding + output_padding;
        if (output_index < 0 || output_index >= output.size(2)) {
            continue;
        }
        scalar_t val = 0;
        for (int k = 0; k < kernel_size; ++k) {
            const int w_idx = in_channel_group * kernel_size + k;
            val += input[batch][in_channel][i] * weight[w_idx][out_channel][k];
        }
        atomicAdd(&output[batch][out_channel_total][output_index], val);
    }
}

at::Tensor conv_transpose1d_forward_cuda(const at::Tensor input, const at::Tensor weight, 
                                        int stride, int padding, int output_padding, 
                                        int kernel_size, int groups) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_length = input.size(2);
    const int out_channels = weight.size(1) * groups;
    const int out_length = (in_length - 1) * stride + kernel_size - 2 * padding + output_padding;

    auto output = at::zeros({batch_size, out_channels, out_length}, input.options());

    const int blocks = batch_size * (in_channels / weight.size(0)) * groups;
    const int threads = weight.size(1);

    auto input_accessor = input.packed_accessor<float,4,torch::RestrictPtrTraits>();
    auto weight_accessor = weight.packed_accessor<float,4,torch::RestrictPtrTraits>();
    auto output_accessor = output.packed_accessor<float,4,torch::RestrictPtrTraits>();

    conv_transpose1d_backward_data_kernel<float><<<dim3(blocks), dim3(threads)>>>(
        input_accessor, weight_accessor, output_accessor,
        stride, padding, output_padding, kernel_size, groups);

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose_1d_cpp_source = """
at::Tensor conv_transpose1d_forward_cuda(const at::Tensor input, const at::Tensor weight, 
                                        int stride, int padding, int output_padding, 
                                        int kernel_size, int groups);
"""

conv_transpose_1d = load_inline(
    name="conv_transpose_1d",
    cpp_sources=conv_transpose_1d_cpp_source,
    cuda_sources=conv_transpose_1d_source,
    functions=["conv_transpose1d_forward_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                stride: int = 1, padding: int = 0, output_padding: int = 0, 
                groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weight similar to PyTorch's ConvTranspose1d
        weight_shape = (in_channels // groups, out_channels // groups, kernel_size, 1)
        self.weight = nn.Parameter(torch.randn(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv_transpose_1d.conv_transpose1d_forward_cuda(
            x.unsqueeze(-1), self.weight.cuda(), 
            self.stride, self.padding, self.output_padding, 
            self.kernel_size, self.groups)
        output = output.squeeze(-1)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
        return output