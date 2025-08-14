import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int stride,
    int padding,
    int dilation
) {
    int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    int output_y = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;

    if (output_x >= output_width || output_y >= output_height || batch >= batch_size)
        return;

    for (int oc = threadIdx.z; oc < out_channels; oc += blockDim.z) {
        float acc = 0.0;
        for (int k_h = 0; k_h < kernel_size; ++k_h) {
            for (int k_w = 0; k_w < kernel_size; ++k_w) {
                // Compute the corresponding input position accounting for dilation and stride
                int in_x = (output_x - padding - k_w * dilation) / stride;
                int in_y = (output_y - padding - k_h * dilation) / stride;

                if ((output_x - padding - k_w * dilation) % stride != 0 ||
                    (output_y - padding - k_h * dilation) % stride != 0) {
                    continue;
                }

                in_x = (output_x - padding - k_w * dilation) / stride;
                in_y = (output_y - padding - k_h * dilation) / stride;

                if (in_x < 0 || in_x >= input_width || in_y < 0 || in_y >= input_height) {
                    continue;
                }

                for (int ic = 0; ic < in_channels; ++ic) {
                    // Weight layout is [oc, ic, kh, kw]
                    const float w = weight[oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + k_h * kernel_size + k_w];
                    const float in_val = input[batch * in_channels * input_height * input_width + ic * input_height * input_width + in_y * input_width + in_x];
                    acc += w * in_val;
                }
            }
        }
        output[batch * out_channels * output_height * output_width + oc * output_height * output_width + output_y * output_width + output_x] = acc;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    // Compute output dimensions using standard transpose convolution formula
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + 2 * padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + 2 * padding;

    torch::Tensor output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 threads(16, 16, 4); // Configure based on architecture
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size
    );

    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        input_height,
        input_width,
        output_height,
        output_width,
        stride,
        padding,
        dilation
    );

    return output;
}
"""

conv_transpose2d_cpp_source = "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"

conv_transpose2d_extension = load_inline(
    name="conv_transpose2d_ext",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        # Initialize weights and bias similar to PyTorch's default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = conv_transpose2d_extension.conv_transpose2d_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output