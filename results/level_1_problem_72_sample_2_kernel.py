import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose3d
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel function for 3D transpose convolution
__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int padding_d, const int padding_h, const int padding_w,
    const int output_padding_d, const int output_padding_h, const int output_padding_w,
    const int groups,
    const int output_depth, const int output_height, const int output_width,
    const int input_depth, const int input_height, const int input_width
) {
    const int output_size = output_depth * output_height * output_width;
    const int channel_group = out_channels / groups;
    const int in_per_group = in_channels / groups;

    // Each thread handles one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    int w_pos = idx % output_width;
    int h_pos = (idx / output_width) % output_height;
    int d_pos = (idx / (output_width * output_height)) % output_depth;
    int b = (idx / (output_depth * output_height * output_width));

    // Compute input position based on deconvolution formula
    int input_d = (d_pos - padding_d - output_padding_d) / stride_d;
    int input_h = (h_pos - padding_h - output_padding_h) / stride_h;
    int input_w = (w_pos - padding_w - output_padding_w) / stride_w;

    // If the input position is out of bounds, this output position isn't contributed to by the input, skip
    if (input_d < 0 || input_d >= input_depth) return;
    if (input_h < 0 || input_h >= input_height) return;
    if (input_w < 0 || input_w >= input_width) return;

    const int output_offset = (b * out_channels + (d_pos * output_height + h_pos) * output_width + w_pos);
    const int input_offset = (b * in_channels + (input_d * input_height + input_h) * input_width + input_w);

    for (int g = 0; g < groups; g++) {
        const int in_group_start = g * in_per_group;
        const int out_group_start = g * channel_group;
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int k_offset = (kd * kernel_h + kh) * kernel_w + kw;
                    const int weight_idx = (out_group_start + (kd * kernel_h + kh) * kernel_w + kw) * in_per_group;
                    for (int ic = 0; ic < in_per_group; ++ic) {
                        const float w = weight[weight_idx + ic];
                        const float in_val = input[input_offset + in_group_start + ic];
                        atomicAdd(&output[output_offset + out_group_start + ic * kernel_d * kernel_h * kernel_w + k_offset], w * in_val);
                    }
                }
            }
        }
    }
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups
) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0) / (kernel_d * kernel_h * kernel_w);

    // Compute output dimensions
    const auto input_depth = input.size(2);
    const auto input_height = input.size(3);
    const auto input_width = input.size(4);

    const auto output_depth = (input_depth - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    const auto output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    const auto output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    const int total_elements = batch_size * output_depth * output_height * output_width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        groups,
        output_depth, output_height, output_width,
        input_depth, input_height, input_width
    );

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups
);
"""

# Compile the inline CUDA code for ConvTranspose3d
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weights similar to PyTorch's ConvTranspose3d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Using same initialization

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unpack parameters
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        opd, op_h, op_w = self.output_padding

        output = conv_transpose3d.conv_transpose3d_cuda(
            x,
            self.weight,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            opd, op_h, op_w,
            self.groups
        )

        if self.bias is not None:
            # Add bias
            bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
            output += bias_view

        return output