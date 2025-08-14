import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv3d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,5> input,
    const torch::PackedTensorAccessor<scalar_t,5> weight,
    torch::PackedTensorAccessor<scalar_t,5> output,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    int batch_size,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width) {

    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int m = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= out_channels || c >= groups || n >= batch_size) return;

    scalar_t sum = 0;
    for (int di = 0; di < kernel_size; ++di) {
        for (int dj = 0; dj < kernel_size; ++dj) {
            for (int dk = 0; dk < kernel_size; ++dk) {
                for (int s = 0; s < (in_channels / groups); ++s) {
                    int input_d = di*dilation + (stride*(m / (out_channels / groups))) % input_depth;
                    int input_h = dj*dilation + (stride*(m % (out_channels / groups) / (output_width))) % input_height;
                    int input_w = dk*dilation + (stride*(m % (output_width))) % input_width;
                    if (input_d < 0 || input_d >= input_depth || input_h <0 || input_h >= input_height || input_w <0 || input_w >= input_width) continue;

                    sum += input[n][c* (in_channels/groups) + s][input_d][input_h][input_w] * 
                        weight[m][c* (in_channels/groups) + s][di][dj][dk];
                }
            }
        }
    }
    output[n][m][...] = sum;
}

torch::Tensor conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Calculate output dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    int kernel_size = weight.size(2);
    int out_channels = weight.size(0);

    int output_depth = (input_depth + 2*padding - dilation*(kernel_size-1) -1)/stride +1;
    int output_height = (input_height + 2*padding - dilation*(kernel_size-1) -1)/stride +1;
    int output_width = (input_width + 2*padding - dilation*(kernel_size-1) -1)/stride +1;

    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    const int threads = 512;
    dim3 blocks(
        (out_channels + threads -1)/threads,
        (groups + threads -1)/threads,
        (batch_size + threads -1)/threads
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_forward", ([&] {
        conv3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5>(),
            weight.packed_accessor<scalar_t,5>(),
            output.packed_accessor<scalar_t,5>(),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            batch_size,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width);
    }));

    return output;
}
"""

conv3d_cpp_source = (
    "torch::Tensor conv3d_forward(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation, int groups);"
)

# Compile the inline CUDA code for 3D convolution
conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward"],
    verbose=True,
    extra_cflags=["-D__CUDA_NO_HALF_OPERATORS__"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        # Initialize weights (simple initialization, replace with proper initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
        self.conv3d = conv3d

    def forward(self, x):
        output = self.conv3d.conv3d_forward(x, self.weight, self.stride, self.padding, self.dilation, self.groups)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        return output