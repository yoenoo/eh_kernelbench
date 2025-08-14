import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.register_buffer('weight', torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.register_buffer('bias', torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Define custom convolution kernel
        conv3d_kernel = f"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); i++)

template <typename scalar_t>
__global__ void custom_conv3d_forward(const scalar_t* __restrict__ input,
                                    const scalar_t* __restrict__ weight,
                                    scalar_t* __restrict__ output,
                                    const int output_depth,
                                    const int output_height,
                                    const int output_width,
                                    const int input_channels,
                                    const int output_channels,
                                    const int kernel_depth,
                                    const int kernel_height,
                                    const int kernel_width,
                                    const int stride_d,
                                    const int stride_h,
                                    const int stride_w,
                                    const int padding_d,
                                    const int padding_h,
                                    const int padding_w,
                                    const int dilation_d,
                                    const int dilation_h,
                                    const int dilation_w,
                                    const int groups) {{
    const int output_size = output_depth * output_height * output_width;
    const int num_kernels = output_channels / groups;
    const int channels_per_group = input_channels / groups;

    CUDA_KERNEL_LOOP(index, output_channels * output_size) {{
        int c_out = index / output_size;
        int spatial = index % output_size;
        int d_out = spatial / (output_height * output_width);
        int h_out = (spatial / output_width) % output_height;
        int w_out = spatial % output_width;

        scalar_t val = 0;
        const int group = c_out / num_kernels;
        const int c_out_in_group = c_out % num_kernels;

        for (int k_depth = 0; k_depth < kernel_depth; k_depth++) {{
            const int d_in = d_out * stride_d - padding_d + k_depth * dilation_d;
            if (d_in < 0 || d_in >= input_depth) continue;

            for (int k_height = 0; k_height < kernel_height; k_height++) {{
                const int h_in = h_out * stride_h - padding_h + k_height * dilation_h;
                if (h_in < 0 || h_in >= input_height) continue;

                for (int k_width = 0; k_width < kernel_width; k_width++) {{
                    const int w_in = w_out * stride_w - padding_w + k_width * dilation_w;
                    if (w_in < 0 || w_in >= input_width) continue;

                    for (int c_in = 0; c_in < channels_per_group; c_in++) {{
                        const int input_offset = ((group * channels_per_group + c_in) * input_depth +
                                                d_in) * input_height * input_width +
                                                h_in * input_width + w_in;
                        const int weight_offset = (c_out_in_group * channels_per_group + c_in) *
                                                kernel_depth * kernel_height * kernel_width +
                                                k_depth * kernel_height * kernel_width +
                                                k_height * kernel_width + k_width;
                        val += input[input_offset] * weight[weight_offset];
                    }}
                }}
            }}
        }}
        if (bias) {{
            val += bias[c_out];
        }}
        output[index] = val;
    }}
}}

torch::Tensor custom_conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                        int output_depth, int output_height, int output_width,
                                        int input_channels, int output_channels, int kernel_depth,
                                        int kernel_height, int kernel_width,
                                        int stride_d, int stride_h, int stride_w,
                                        int padding_d, int padding_h, int padding_w,
                                        int dilation_d, int dilation_h, int dilation_w,
                                        int groups) {{
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);

    auto output = torch::empty({{input.size(0), output_channels, output_depth, output_height, output_width}}, 
                                input.options());

    dim3 blocks(TORCH_CUDA_GET_BLOCKS(output.size(0)*output.size(1)*output.size(2)*output.size(3)*output.size(4)));
    dim3 threads(TORCH_CUDA_THREADS);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv3d_forward", ([&] {{        
        custom_conv3d_forward<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            output_depth,
            output_height,
            output_width,
            input_channels,
            output_channels,
            kernel_depth,
            kernel_height,
            kernel_width,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            dilation_d,
            dilation_h,
            dilation_w,
            groups
        );
    }));

    if (bias.defined()) {{
        output += bias.view(1, -1, 1, 1, 1);
    }}
    return output;
}}
"""
        # Compile kernel
        self.custom_conv3d = load_inline(
            name='custom_conv3d',
            cpp_sources=conv3d_kernel,
            functions=['custom_conv3d_forward_cuda'],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate output dimensions
        input_depth = x.size(2)
        input_height = x.size(3)
        input_width = x.size(4)
        kernel_d, kernel_h, kernel_w = self.weight.size()[2:]
        output_depth = (input_depth + 2*self.padding[0] - self.dilation[0]*(kernel_d - 1) - 1)//self.stride[0] + 1
        output_height = (input_height + 2*self.padding[1] - self.dilation[1]*(kernel_h - 1) - 1)//self.stride[1] + 1
        output_width = (input_width + 2*self.padding[2] - self.dilation[2]*(kernel_w - 1) - 1)//self.stride[2] + 1

        # Run custom convolution
        return self.custom_conv3d.custom_conv3d_forward_cuda(
            x, 
            self.weight, 
            self.bias if hasattr(self, 'bias') else torch.empty(0),
            output_depth,
            output_height,
            output_width,
            x.size(1),
            self.weight.size(0),
            kernel_d,
            kernel_h,
            kernel_w,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.dilation[0],
            self.dilation[1],
            self.dilation[2],
            self.groups
        )