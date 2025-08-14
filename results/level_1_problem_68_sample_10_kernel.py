import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose3d
conv_transpose3d_source = """
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_depth,
    const int kernel_width,
    const int kernel_height,
    const int stride_depth,
    const int stride_width,
    const int stride_height,
    const int padding_depth,
    const int padding_width,
    const int padding_height,
    const int output_padding_depth,
    const int output_padding_width,
    const int output_padding_height,
    const int input_depth,
    const int input_width,
    const int input_height,
    const int output_depth,
    const int output_width,
    const int output_height
) {
    CUDA_1D_KERNEL_LOOP(output_idx, batch_size * out_channels * output_depth * output_width * output_height) {
        int b = output_idx / (out_channels * output_depth * output_width * output_height);
        int c_out = (output_idx / (output_depth * output_width * output_height)) % out_channels;
        int d_out = (output_idx / (output_width * output_height)) % output_depth;
        int w_out = (output_idx / output_height) % output_width;
        int h_out = output_idx % output_height;

        const int d_in_start = (d_out + padding_depth - output_padding_depth) / stride_depth;
        const int w_in_start = (w_out + padding_width - output_padding_width) / stride_width;
        const int h_in_start = (h_out + padding_height - output_padding_height) / stride_height;

        scalar_t val = 0;
        for (int k_d = 0; k_d < kernel_depth; ++k_d) {
            const int d_in = d_in_start - k_d;
            if (d_in < 0 || d_in >= input_depth) continue;

            for (int k_w = 0; k_w < kernel_width; ++k_w) {
                const int w_in = w_in_start - k_w;
                if (w_in < 0 || w_in >= input_width) continue;

                for (int k_h = 0; k_h < kernel_height; ++k_h) {
                    const int h_in = h_in_start - k_h;
                    if (h_in < 0 || h_in >= input_height) continue;

                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        const int input_offset = b * in_channels * input_depth * input_width * input_height +
                                                c_in * input_depth * input_width * input_height +
                                                d_in * input_width * input_height +
                                                w_in * input_height +
                                                h_in;

                        const int weight_offset = c_out * in_channels * kernel_depth * kernel_width * kernel_height +
                                                c_in * kernel_depth * kernel_width * kernel_height +
                                                k_d * kernel_width * kernel_height +
                                                k_w * kernel_height +
                                                k_h;

                        val += input[input_offset] * weight[weight_offset];
                    }
                }
            }
        }
        output[output_idx] = val;
    }
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight,
    int kernel_depth, int kernel_width, int kernel_height,
    int stride_depth, int stride_width, int stride_height,
    int padding_depth, int padding_width, int padding_height,
    int output_padding_depth, int output_padding_width, int output_padding_height,
    int output_depth, int output_width, int output_height) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int input_depth = input.size(2);
    const int input_width = input.size(3);
    const int input_height = input.size(4);

    auto output = torch::zeros({batch_size, out_channels, output_depth, output_width, output_height}, input.options());

    dim3 blocks = dim3(ceil( (output.numel()) / 256.0 ));
    dim3 threads = dim3(256);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_cuda", ([&] {
        conv_transpose3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_depth,
            kernel_width,
            kernel_height,
            stride_depth,
            stride_width,
            stride_height,
            padding_depth,
            padding_width,
            padding_height,
            output_padding_depth,
            output_padding_width,
            output_padding_height,
            input_depth,
            input_width,
            input_height,
            output_depth,
            output_width,
            output_height);
    }));

    return output;
}

"""

# Compile the inline CUDA code
conv_transpose3d_cpp_source = (
    "torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight,"
    "int kernel_depth, int kernel_width, int kernel_height,"
    "int stride_depth, int stride_width, int stride_height,"
    "int padding_depth, int padding_width, int padding_height,"
    "int output_padding_depth, int output_padding_width, int output_padding_height,"
    "int output_depth, int output_width, int output_height);"
)

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA", "-O3"],
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
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) 

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate output shapes using same logic as ConvTranspose3d
        input_depth = x.size(2)
        input_width = x.size(3)
        input_height = x.size(4)

        # Compute output dimensions
        output_depth = (input_depth - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0] + self.output_padding[0]
        output_width = (input_width - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1] + self.output_padding[1]
        output_height = (input_height - 1) * self.stride[2] - 2 * self.padding[2] + self.kernel_size[2] + self.output_padding[2]

        # Run custom CUDA kernel
        result = conv_transpose3d.conv_transpose3d_cuda(
            x,
            self.weight,
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2],
            output_depth, output_width, output_height
        )

        if self.bias is not None:
            result += self.bias.view(1, -1, 1, 1, 1)

        return result