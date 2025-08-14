import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, int).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weights similar to PyTorch's default initialization
        weight_shape = (in_channels, out_channels // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            bias_shape = (out_channels,)
            self.bias = nn.Parameter(torch.empty(bias_shape))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        # Load custom CUDA kernel
        self.cuda_conv_trans = self.load_conv_transpose2d_kernel()

    def load_conv_transpose2d_kernel(self):
        kernel_code = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <vector>

        __global__ void conv_transpose2d_kernel(
            const float* input,
            const float* weight,
            float* output,
            const int batch_size,
            const int in_channels,
            const int out_channels,
            const int kernel_size,
            const int stride,
            const int padding,
            const int output_padding,
            const int groups,
            const int input_height,
            const int input_width,
            const int output_height,
            const int output_width) {

            // Implementation details for transposed convolution
            // Note: This is a simplified version and may require tuning for performance.
            // The exact indices and loop structure would depend on the specific parameters and optimization strategy.
            const int h_out = blockIdx.y;
            const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
            const int group = blockIdx.z;

            if (w_out >= output_width) return;

            const int in_channels_per_group = in_channels / groups;
            const int out_channels_per_group = out_channels / groups;

            float val = 0.0;
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int h_in = (h_out * stride - padding) - kh;
                    int w_in = (w_out * stride - padding) - kw;
                    if (h_in < 0 || w_in < 0 || h_in >= input_height || w_in >= input_width) {
                        continue;
                    }
                    for (int ich = 0; ich < in_channels_per_group; ++ich) {
                        for (int och = 0; och < out_channels_per_group; ++och) {
                            val += input[batch_idx * in_channels * input_height * input_width +
                                        (group * in_channels_per_group + ich) * input_height * input_width +
                                        h_in * input_width + w_in] *
                                   weight[(group * out_channels_per_group + och) * kernel_size * kernel_size * in_channels_per_group +
                                          ich * kernel_size * kernel_size +
                                          kh * kernel_size + kw];
                        }
                    }
                }
            }
            output[...] = val; // Assign to correct output indices
        }

        torch::Tensor conv_transpose2d(
            torch::Tensor input,
            torch::Tensor weight,
            int stride,
            int padding,
            int output_padding,
            int groups) {

            // Compute output dimensions
            const int batch_size = input.size(0);
            const int in_channels = input.size(1);
            const int input_height = input.size(2);
            const int input_width = input.size(3);
            const int out_channels = weight.size(0) * groups;
            const int kernel_size = weight.size(2);
            const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
            const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

            auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

            dim3 threads(256);
            dim3 blocks(div_up(output_width, threads.x), output_height, groups);

            conv_transpose2d_kernel<<<blocks, threads>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                output_padding,
                groups,
                input_height,
                input_width,
                output_height,
                output_width);
            return output;
        }
        """

        return load_inline(
            name="conv_transpose2d",
            cpp_sources="torch::Tensor conv_transpose2d(torch::Tensor input, torch::Tensor weight, int stride, int padding, int output_padding, int groups);",
            cuda_sources=kernel_code,
            functions=["conv_transpose2d"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.cuda_conv_trans.conv_transpose2d(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output

def div_up(a, b):
    return (a + b - 1) // b