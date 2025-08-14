import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from torch import nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define custom kernel parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize the parameters
        self.reset_parameters()
        
        # Load the custom CUDA kernel
        self.conv3d_transpose_kernel = load_inline(
            name='conv3d_transpose_cuda',
            cuda Sources=self.get_conv3d_transpose_source(),
            functions=['conv_transpose3d_cuda'],
            verbose=True
        )

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def get_conv3d_transpose_source(self):
        return """
            #include <torch/extension.h>
            #include <cuda.h>
            #include <cuda_runtime.h>
            #include <vector>

            template <typename scalar_t>
            __global__ void conv3d_transpose_kernel(const scalar_t* __restrict__ input,
                                                    const scalar_t* __restrict__ weight,
                                                    scalar_t* output,
                                                    int batch_size, int in_channels, int depth, int width, int height,
                                                    int out_channels, int kernel_depth, int kernel_width, int kernel_height,
                                                    int stride_d, int stride_h, int stride_w,
                                                    int padding_d, int padding_h, int padding_w,
                                                    int output_padding_d, int output_padding_h, int output_padding_w,
                                                    int groups) {

                int batch_idx = blockIdx.x;
                int out_d = blockIdx.y;
                int out_h = threadIdx.y;
                int out_w = threadIdx.x;

                scalar_t val = 0;

                for (int g = 0; g < groups; ++g) {
                    int in_group = in_channels / groups;
                    int out_group = out_channels / groups;

                    int in_c_start = g * in_group;
                    int out_c_start = g * out_group;

                    for (int kernel_d = 0; kernel_d < kernel_depth; ++kernel_d) {
                        for (int kernel_h = 0; kernel_h < kernel_height; ++kernel_h) {
                            for (int kernel_w = 0; kernel_w < kernel_width; ++kernel_w) {

                                int in_d = out_d * stride_d - padding_d - kernel_d;
                                int in_h = out_h * stride_h - padding_h - kernel_h;
                                int in_w = out_w * stride_w - padding_w - kernel_w;

                                if (in_d < 0 || in_h < 0 || in_w < 0) continue;

                                for (int in_c = in_c_start; in_c < in_c_start + in_group; ++in_c) {
                                    for (int out_c = out_c_start; out_c < out_c_start + out_group; ++out_c) {

                                        int weight_offset = out_c * in_group * kernel_depth * kernel_height * kernel_width +
                                                            in_c * kernel_depth * kernel_height * kernel_width +
                                                            kernel_d * kernel_height * kernel_width +
                                                            kernel_h * kernel_width +
                                                            kernel_w;

                                        int input_offset = batch_idx * in_channels * depth * width * height +
                                                            in_c * depth * width * height +
                                                            in_d * width * height +
                                                            in_h * height +
                                                            in_w;

                                        val += weight[weight_offset] * input[input_offset];
                                    }
                                }
                            }
                        }
                    }
                }

                int out_depth = (depth - 1) * stride_d - 2 * padding_d + kernel_depth + output_padding_d;
                int out_height = (height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;
                int out_width = (width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;

                int output_offset = batch_idx * out_channels * out_depth * out_height * out_width +
                                    out_c * out_depth * out_height * out_width +
                                    out_d * out_height * out_width +
                                    out_h * out_width +
                                    out_w;

                output[output_offset] = val;
            }

            torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                                int batch_size, int in_channels, int depth, int width, int height,
                                                int out_channels, int kernel_depth, int kernel_width, int kernel_height,
                                                int stride_d, int stride_h, int stride_w,
                                                int padding_d, int padding_h, int padding_w,
                                                int output_padding_d, int output_padding_h, int output_padding_w,
                                                int groups) {

                const auto output_depth = (depth - 1) * stride_d - 2 * padding_d + kernel_depth + output_padding_d;
                const auto output_height = (height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;
                const auto output_width = (width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;

                auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

                dim3 threads(32, 32);
                dim3 blocks(batch_size, output_depth);

                AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_transpose_cuda", ([&] {
                    conv3d_transpose_kernel<scalar_t><<<blocks, threads>>>(
                        input.data<scalar_t>(),
                        weight.data<scalar_t>(),
                        output.data<scalar_t>(),
                        batch_size, in_channels, depth, width, height,
                        out_channels, kernel_depth, kernel_width, kernel_height,
                        stride_d, stride_h, stride_w,
                        padding_d, padding_h, padding_w,
                        output_padding_d, output_padding_h, output_padding_w,
                        groups
                    );
                }));

                if (bias.defined()) {
                    output += bias.view(1, -1, 1, 1, 1);
                }

                return output;
            }
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract dimensions and parameters
        batch_size, in_channels, depth, width, height = x.size()
        kernel_depth, kernel_width, kernel_height = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        output_padding_d, output_padding_h, output_padding_w = self.output_padding

        # Execute custom kernel
        output = self.conv3d_transpose_kernel.conv_transpose3d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            self.bias if self.bias is not None else torch.Tensor(),
            batch_size,
            in_channels,
            depth,
            width,
            height,
            self.out_channels,
            kernel_depth,
            kernel_width,
            kernel_height,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            output_padding_d,
            output_padding_h,
            output_padding_w,
            self.groups
        )

        return output