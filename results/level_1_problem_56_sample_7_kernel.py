import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized 2D convolution
conv2d_custom_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv2d_kernel(const torch::PackedTensorAccessor<scalar_t,4> input,
                            const torch::PackedTensorAccessor<scalar_t,4> weight,
                            torch::PackedTensorAccessor<scalar_t,4> output,
                            int batch_size,
                            int in_channels,
                            int out_channels,
                            int kernel_h,
                            int kernel_w,
                            int in_h,
                            int in_w,
                            int stride_h,
                            int stride_w,
                            int padding_h,
                            int padding_w,
                            int dilation_h,
                            int dilation_w,
                            int groups) {
    // Calculate output dimensions
    const int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    const int output_channel = blockIdx.z * groups + threadIdx.z;
    const int n = blockIdx.x;
    const int out_x = blockIdx.y * blockDim.y + threadIdx.y;

    if (output_channel >= out_channels || out_x >= out_w) return;

    const int in_channel_group = output_channel / groups;
    const int group_id = output_channel % groups;

    int sum = 0;
    for (int kernel_pos = 0; kernel_pos < kernel_h * kernel_w; ++kernel_pos++) {
        const int kernel_y = kernel_pos / kernel_w;
        const int kernel_x = kernel_pos % kernel_w;

        const int in_y = -padding_h + kernel_y * dilation_h;
        const int in_x = -padding_w + kernel_x * dilation_w;

        if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
            const int input_channel = in_channel_group * in_channels / groups;
            const int output_channel_base = group_id * (out_channels / groups);
            const int weight_idx = output_channel_base + output_channel;

            sum += input[n][input_channel][in_y][in_x] * weight[weight_idx][input_channel][kernel_y][kernel_x];
        }
    }

    output[n][output_channel][0][out_x] = sum;
}

torch::Tensor conv2d_custom_cuda(torch::Tensor input, torch::Tensor weight,
                                int stride_h, int stride_w,
                                int padding_h, int padding_w,
                                int dilation_h, int dilation_w,
                                int groups) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels = weight.size(0);

    // Compute output dimensions
    const int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Define output tensor
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

    dim3 threads(32, 8, 1); // Adjust thread configuration based on input size and kernel
    dim3 blocks(batch_size, (out_w + threads.y - 1)/threads.y, out_channels);

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_kernel", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            batch_size, in_channels, out_channels,
            kernel_h, kernel_w, in_h, in_w,
            stride_h, stride_w, padding_h, padding_w,
            dilation_h, dilation_w, groups);
    }));

    return output;
}
"""

# Declare the C++ function prototype
conv2d_custom_header = """
torch::Tensor conv2d_custom_cuda(torch::Tensor input,
                                torch::Tensor weight,
                                int stride_h, int stride_w,
                                int padding_h, int padding_w,
                                int dilation_h, int dilation_w,
                                int groups);
"""

# Compile the custom convolution kernel
conv2d_custom = load_inline(
    name="conv2d_custom",
    cpp_sources=conv2d_custom_header,
    cuda_sources=conv2d_source,
    functions=["conv2d_custom_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights similar to PyTorch's Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        # Reference to the custom kernel function
        self.conv2d_custom = conv2d_custom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv2d_custom.conv2d_custom_cuda(
            x,
            self.weight,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)  # Add bias if present
        return output