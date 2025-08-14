import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void ConvTranspose3DKernel(const scalar_t* input,
                                     const scalar_t* weight,
                                     scalar_t* output,
                                     int batch_size,
                                     int in_channels,
                                     int out_channels,
                                     int input_depth,
                                     int input_height,
                                     int input_width,
                                     int kernel_size,
                                     int stride,
                                     int padding,
                                     int dilation) {

    int output_depth = (input_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= batch_size * out_channels * output_depth * output_height * output_width) return;

    int w = n % output_width;
    int h = (n / output_width) % output_height;
    int d = (n / (output_width * output_height)) % output_depth;
    int c_out = (n / (output_width * output_height * output_depth)) % out_channels;
    int b = n / (out_channels * output_depth * output_height * output_width);

    scalar_t val = 0;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int k_d = 0; k_d < kernel_size; ++k_d) {
            for (int k_h = 0; k_h < kernel_size; ++k_h) {
                for (int k_w = 0; k_w < kernel_size; ++k_w) {
                    // Compute input coordinates
                    int in_d = (d - dilation * k_d + 2 * padding) / stride;
                    int in_h = (h - dilation * k_h + 2 * padding) / stride;
                    int in_w = (w - dilation * k_w + 2 * padding) / stride;

                    // Check if input coordinates are valid
                    if (in_d < 0 || in_d >= input_depth ||
                        in_h < 0 || in_h >= input_height ||
                        in_w < 0 || in_w >= input_width) {
                        continue;
                    }

                    // Check if the current position is aligned with the stride
                    if ((d - dilation * k_d + 2 * padding) % stride != 0 ||
                        (h - dilation * k_h + 2 * padding) % stride != 0 ||
                        (w - dilation * k_w + 2 * padding) % stride != 0) {
                        continue;
                    }

                    int weight_offset = c_out * in_channels * kernel_size * kernel_size * kernel_size +
                                       c_in * kernel_size * kernel_size * kernel_size +
                                       k_d * kernel_size * kernel_size +
                                       k_h * kernel_size +
                                       k_w;

                    int input_offset = b * in_channels * input_depth * input_height * input_width +
                                      c_in * input_depth * input_height * input_width +
                                      in_d * input_height * input_width +
                                      in_h * input_width +
                                      in_w;

                    val += input[input_offset] * weight[weight_offset];
                }
            }
        }
    }

    int output_offset = b * out_channels * output_depth * output_height * output_width +
                       c_out * output_depth * output_height * output_width +
                       d * output_height * output_width +
                       h * output_width +
                       w;

    output[output_offset] = val;
}

torch::Tensor conv_transpose_3d_cuda(torch::Tensor input,
                                    torch::Tensor weight,
                                    int stride,
                                    int padding,
                                    int dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);
    const int kernel_size = weight.size(2); // Assuming kernel dimensions are same for all spatial dims
    const int out_channels = weight.size(0);

    // Compute output shape
    int output_depth = (input_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, output_options);

    int num_threads = batch_size * out_channels * output_depth * output_height * output_width;
    int block_size = 256;
    int grid_size = (num_threads + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose_3d_cuda", ([&] {
        ConvTranspose3DKernel<scalar_t><<<grid_size, block_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_depth,
            input_height,
            input_width,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose_3d_cpp_source = (
    "torch::Tensor conv_transpose_3d_cuda(torch::Tensor input,"
    "                                    torch::Tensor weight,"
    "                                    int stride,"
    "                                    int padding,"
    "                                    int dilation);"
)

# Compile the inline CUDA code for 3D transposed convolution
conv_transpose_3d = load_inline(
    name="conv_transpose_3d",
    cpp_sources=conv_transpose_3d_cpp_source,
    cuda_sources=conv_transpose_3d_source,
    functions=["conv_transpose_3d_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize weight and bias similar to PyTorch's ConvTranspose3d
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = conv_transpose_3d.conv_transpose_3d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.dilation
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        return output