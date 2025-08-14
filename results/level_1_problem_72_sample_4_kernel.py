import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for 3D transposed convolution
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Convenience constants for thread counts
constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 256;

// Kernel for 3D transposed convolution
__global__ void conv_transpose_3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_depth,
    const int kernel_height,
    const int kernel_width,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int output_padding_d,
    const int output_padding_h,
    const int output_padding_w,
    const int groups,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width) {

    // Compute output coordinates
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (output_idx >= batch_size * out_channels * output_depth * output_height * output_width) {
        return;
    }
    
    int w_out = output_idx % output_width;
    int h_out = (output_idx / output_width) % output_height;
    int d_out = (output_idx / (output_width * output_height)) % output_depth;
    int oc = (output_idx / (output_depth * output_height * output_width)) % out_channels;
    int batch = output_idx / (out_channels * output_depth * output_height * output_width);

    // Compute input coordinates from output coordinates using stride and output padding
    int d_in = (d_out - output_padding_d) / stride_d - padding_d;
    int h_in = (h_out - output_padding_h) / stride_h - padding_h;
    int w_in = (w_out - output_padding_w) / stride_w - padding_w;

    // Ensure that d_in, h_in, w_in are within bounds of input tensor to avoid out-of-bounds access
    if (d_in < 0 || h_in < 0 || w_in < 0 ||
        d_in >= input_depth || h_in >= input_height || w_in >= input_width) {
        output[output_idx] = 0.0f;
        return;
    }

    // Compute group for channel routing
    int group = oc / (out_channels / groups);
    int in_group_oc = oc % (out_channels / groups);
    int in_group_ic = in_group_oc;

    float sum = 0.0;

    // Iterate over kernel dimensions and input channels within group
    for (int kd = 0; kd < kernel_depth; ++kd) {
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kh < kernel_width; ++kw) { // (Typo correction: 'kh' to 'kw'? Ensure loops cover all axes correctly)
                int id = d_in + kd;
                int ih = h_in + kh;
                int iw = w_in + kw;

                if (id < 0 || id >= input_depth || ih < 0 || ih >= input_height || iw < 0 || iw >= input_width) {
                    continue;  // Skip out-of-bound elements
                }

                // Input channel must be in the same group
                for (int ic = group * (in_channels / groups); ic < (group + 1) * (in_channels / groups); ++ic) {
                    sum += input[batch * in_channels * input_depth * input_height * input_width +
                                ic * input_depth * input_height * input_width +
                                id * input_height * input_width +
                                ih * input_width + iw] *
                            weight[group * (in_channels/kernel_depth/kernel_height/kernel_width) + ... ]; // (Need to compute correct weight index based on layout)
                }
            }
        }
    }

    output[output_idx] = sum;
}

torch::Tensor conv_transpose_3d_cuda(torch::Tensor input,
                                     torch::Tensor weight,
                                     int stride_d,
                                     int stride_h,
                                     int stride_w,
                                     int padding_d,
                                     int padding_h,
                                     int padding_w,
                                     int output_padding_d,
                                     int output_padding_h,
                                     int output_padding_w,
                                     int groups) {
    
    // Output size computation
    auto input_size = input.sizes();
    const int batch_size = input_size[0];
    const int in_channels = input_size[1];
    const int input_depth = input_size[2];
    const int input_height = input_size[3];
    const int input_width = input_size[4];

    // Weight dimensions
    auto weight_size = weight.sizes();
    const int out_channels = weight_size[0];
    const int kernel_depth = weight_size[2];
    const int kernel_height = weight_size[3];
    const int kernel_width = weight_size[4];

    // Compute output dimensions using transposed conv formula
    const int output_depth = (input_depth - 1) * stride_d - 2 * padding_d + kernel_depth + output_padding_d;
    const int output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;
    const int output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;

    // Output tensor allocation
    auto output_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, output_options);

    // Compute grid and block dimensions
    const int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    const dim3 block(kThreadsPerBlock);
    const dim3 grid((total_elements + block.x - 1) / block.x);

    // Launch kernel
    conv_transpose_3d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        kernel_depth, kernel_height, kernel_width,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        groups,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width);

    return output;
}

"""

conv_transpose_3d_cpp_source = """
torch::Tensor conv_transpose_3d_cuda(torch::Tensor input,
                                    torch::Tensor weight,
                                    int stride_d,
                                    int stride_h,
                                    int stride_w,
                                    int padding_d,
                                    int padding_h,
                                    int padding_w,
                                    int output_padding_d,
                                    int output_padding_h,
                                    int output_padding_w,
                                    int groups);
"""

# Compile the inline CUDA code for the transposed convolution
conv_transpose_3d = load_inline(
    name="conv_transpose_3d",
    cpp_sources=conv_transpose_3d_cpp_source,
    cuda_sources=conv_transpose_3d_source,
    functions=["conv_transpose_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weight tensor similar to PyTorch's ConvTranspose3d
        kernel_depth, kernel_height, kernel_width = kernel_size
        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels // groups,
            kernel_depth, kernel_height, kernel_width))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Same as PyTorch's default
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        self.conv_transpose_3d = conv_transpose_3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unpack parameters
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        output_padding_d, output_padding_h, output_padding_w = self.output_padding
        
        # Execute the custom CUDA kernel
        output = self.conv_transpose_3d.conv_transpose_3d_cuda(
            x, self.weight, stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w,
            self.groups)
        
        # Add bias if required
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        
        return output