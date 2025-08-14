import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void ConvTranspose3DForwardKernel(const scalar_t* __restrict__ input,
                                            const scalar_t* __restrict__ weight,
                                            scalar_t* __restrict__ output,
                                            const int batch_size,
                                            const int in_channels,
                                            const int out_channels,
                                            const int kernel_depth,
                                            const int kernel_width,
                                            const int kernel_height,
                                            const int input_depth,
                                            const int input_width,
                                            const int input_height,
                                            const int output_depth,
                                            const int output_width,
                                            const int output_height,
                                            const int stride_d,
                                            const int stride_w,
                                            const int stride_h,
                                            const int padding_d,
                                            const int padding_w,
                                            const int padding_h,
                                            const int output_padding_d,
                                            const int output_padding_w,
                                            const int output_padding_h,
                                            const int groups) {
    // Compute output position (n, c_out, d, w, h)
    const int d_out = blockIdx.z;
    const int w_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int h_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int c_out = threadIdx.z;
    const int n = blockIdx.x * gridDim.y * blockDim.x * gridDim.z * blockDim.y + 
                 blockIdx.y * blockDim.y * blockDim.x * gridDim.z + 
                 blockIdx.z * blockDim.x * blockDim.y + 
                 threadIdx.z;

    if (w_out >= output_width || h_out >= output_height || c_out >= out_channels || n >= batch_size) {
        return;
    }

    scalar_t val = 0;
    const int in_c_group = in_channels / groups;
    const int out_c_group = out_channels / groups;
    const int group_id = c_out / out_c_group;

    for (int kd = 0; kd < kernel_depth; ++kd) {
        for (int kw = 0; kw < kernel_width; ++kw) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                // Compute input coordinates
                const int d_in = (d_out - kd - padding_d + output_padding_d) / stride_d;
                const int w_in = (w_out - kw - padding_w + output_padding_w) / stride_w;
                const int h_in = (h_out - kh - padding_h + output_padding_h) / stride_h;

                if (d_in < 0 || d_in >= input_depth || w_in < 0 || w_in >= input_width || h_in < 0 || h_in >= input_height) {
                    continue;
                }

                // Compute input channel index
                const int in_channel = group_id * in_c_group + ((kd * kernel_width + kw) * kernel_height + kh) * out_channels;

                val += input[n * in_channels * input_depth * input_width * input_height + 
                            group_id * in_c_group + 
                            in_channel * input_depth * input_width * input_height + 
                            d_in * input_width * input_height + 
                            w_in * input_height + h_in] * 
                        weight[c_out * kernel_depth * kernel_width * kernel_height + 
                               kd * kernel_width * kernel_height + 
                               kw * kernel_height + kh];
            }
        }
    }

    const int output_offset = n * out_channels * output_depth * output_width * output_height +
                             c_out * output_depth * output_width * output_height +
                             d_out * output_width * output_height +
                             w_out * output_height + h_out;

    atomicAdd(&output[output_offset], val);
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int kernel_depth,
                                   int kernel_width,
                                   int kernel_height,
                                   int stride_d,
                                   int stride_w,
                                   int stride_h,
                                   int padding_d,
                                   int padding_w,
                                   int padding_h,
                                   int output_padding_d,
                                   int output_padding_w,
                                   int output_padding_h,
                                   int groups) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_depth = input.size(2);
    const int input_width = input.size(3);
    const int input_height = input.size(4);

    const int out_channels = weight.size(0) / kernel_depth / kernel_width / kernel_height;
    const int output_depth = (input_depth - 1) * stride_d - 2 * padding_d + kernel_depth + output_padding_d;
    const int output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;
    const int output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({batch_size, out_channels, output_depth, output_width, output_height}, output_options);

    const dim3 threads(16, 4, 1); // Xavier HBM occupancy max at 16x4 threads
    const dim3 blocks((output_height + threads.x - 1)/threads.x,
                     (output_width + threads.y - 1)/threads.y,
                     output_depth);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_cuda", ([&] {
        ConvTranspose3DForwardKernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, out_channels,
            kernel_depth, kernel_width, kernel_height,
            input_depth, input_width, input_height,
            output_depth, output_width, output_height,
            stride_d, stride_w, stride_h,
            padding_d, padding_w, padding_h,
            output_padding_d, output_padding_w, output_padding_h,
            groups);
    }));

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int kernel_depth,
                                   int kernel_width,
                                   int kernel_height,
                                   int stride_d,
                                   int stride_w,
                                   int stride_h,
                                   int padding_d,
                                   int padding_w,
                                   int padding_h,
                                   int output_padding_d,
                                   int output_padding_w,
                                   int output_padding_h,
                                   int groups);
"""

conv_transpose3d_ops = load_inline(
    name='conv_transpose3d_cuda',
    cpp_sources=[conv_transpose3d_cpp_source],
    cuda_sources=[conv_transpose3d_source],
    functions=['conv_transpose3d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=(1, 1, 1), padding=(0, 0, 0),
                 output_padding=(0, 0, 0), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weights and bias similar to nn.ConvTranspose3d
        kernel_depth, kernel_width, kernel_height = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_depth, kernel_width, kernel_height))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Load CUDA kernel
        self.conv_transpose3d_cuda_op = conv_transpose3d_ops.conv_transpose3d_cuda

        # Initialize weights with same method as PyTorch's ConvTranspose3d
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Extract parameters
        kernel_depth, kernel_width, kernel_height = self.kernel_size
        stride_d, stride_w, stride_h = self.stride
        padding_d, padding_w, padding_h = self.padding
        op_d, op_w, op_h = self.output_padding

        output = self.conv_transpose3d_cuda_op(
            x, self.weight, kernel_depth, kernel_width, kernel_height,
            stride_d, stride_w, stride_h,
            padding_d, padding_w, padding_h,
            op_d, op_w, op_h,
            self.groups
        )

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)

        return output