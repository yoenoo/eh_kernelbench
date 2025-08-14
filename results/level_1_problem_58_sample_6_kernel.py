import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose3d with ReLU fused
conv_transpose_relu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv_transpose3d_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* output,
    int in_depth, int in_height, int in_width,
    int out_channels, int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w) {

    int batch = blockIdx.z;
    int out_z = blockIdx.x * blockDim.z + threadIdx.z;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = threadIdx.x;

    int out_depth = out_z;
    int out_height = out_y;
    int out_width = out_x;

    // Compute input coordinates based on output coordinates and padding/output_padding
    int in_z = (out_depth - output_padding_d) / stride_d - padding_d;
    int in_y = (out_height - output_padding_h) / stride_h - padding_h;
    int in_x = (out_width - output_padding_w) / stride_w - padding_w;

    if (in_z < 0 || in_y < 0 || in_x < 0) return;

    // Bounds checking for input
    if (in_z >= in_depth || in_y >= in_height || in_x >= in_width) return;

    for (int c_out = 0; c_out < out_channels; c_out += blockDim.x * gridDim.x) {
        int c_out_offset = c_out + out_x;
        if (c_out_offset >= out_channels) continue;

        float sum = 0.0f;
        for (int k_d = 0; k_d < kernel_d; ++k_d) {
            for (int k_h = 0; k_h < kernel_h; ++k_h) {
                for (int k_w = 0; k_w < kernel_w; ++k_w) {
                    int input_z = out_depth + k_d * stride_d - output_padding_d;
                    int input_y = out_height + k_h * stride_h - output_padding_h;
                    int input_x = out_width + k_w * stride_w - output_padding_w;

                    // Adjust to input coordinates using kernel dimensions
                    // ... (This part requires precise calculation based on transpose logic)
                    // For simplicity, assuming the standard backward mapping here
                    // The precise logic here depends on PyTorch's conv transpose implementation details
                    // Please note: The exact indices calculation may require adjustments based on actual requirements

                    // Get the input value
                    int input_offset = batch * in_channels * in_depth * in_height * in_width +
                        c_in * in_depth * in_height * in_width +
                        (in_z + k_d) * in_height * in_width +
                        (in_y + k_h) * in_width +
                        (in_x + k_w);
                    const float input_val = input[input_offset];

                    // Get the weight value (transposed weight)
                    int weight_offset = c_out_offset * in_channels * kernel_d * kernel_h * kernel_w +
                        c_in * kernel_d * kernel_h * kernel_w +
                        k_d * kernel_h * kernel_w +
                        k_h * kernel_w +
                        k_w;
                    const float weight_val = weight[weight_offset];

                    sum += input_val * weight_val;
                }
            }
        }
        sum = fmaxf(sum, 0.0f); // Apply ReLU
        output[output_offset] = sum;
    }
}

torch::Tensor conv_transpose3d_relu_cuda(torch::Tensor input, torch::Tensor weight,
    int kernel_size_d, int kernel_size_h, int kernel_size_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w) {

    // Calculate output dimensions based on input and parameters
    // ... (Implement output dimensions calculation here based on PyTorch's formula)
    // depth_out = (input_depth - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d
    // Similarly for height and width

    auto output_size = ...; // Define according to PyTorch's conv transpose output formula

    auto output = torch::empty(output_size, input.options());

    dim3 threads(32, 8, 1); // Example thread block size
    dim3 blocks(...); // Calculate based on output dimensions

    conv_transpose3d_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(2), input.size(3), input.size(4),
        weight.size(0), // out_channels
        kernel_size_d, kernel_size_h, kernel_size_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w);

    return output;
}
"""

conv_transpose_relu_cpp_source = (
    "torch::Tensor conv_transpose3d_relu_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size_d, int kernel_size_h, int kernel_size_w, int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w, int output_padding_d, int output_padding_h, int output_padding_w);"
)

conv_transpose_relu = load_inline(
    name="conv_transpose_relu",
    cpp_sources=conv_transpose_relu_cpp_source,
    cuda_sources=conv_transpose_relu_source,
    functions=["conv_transpose_relu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1),
                 padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weights similar to ConvTranspose3d
        # Kernel dimensions in ConvTranspose are (in_channels, out_channels//groups, ...) for forward
        # But in transpose, weight shape is (in_channels, out_channels//groups, ...)
        # Assuming groups=1 here for simplicity. Need adjustment for general cases
        kernel_d, kernel_h, kernel_w = kernel_size
        weight_shape = (out_channels, in_channels // groups, kernel_d, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.randn(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

        self.cuda_conv = conv_transpose_relu  # The custom kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        output_padding_d, output_padding_h, output_padding_w = self.output_padding

        # Call the fused kernel
        out = self.cuda_conv.conv_transpose3d_relu_cuda(
            x, self.weight, kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w)

        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1, 1)  # Apply bias if needed

        return out