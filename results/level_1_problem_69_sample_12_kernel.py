import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

template <typename scalar_t>
__global__ void conv_transpose2d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int out_channels, int in_channels, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w, int groups) {

    // Implement the transpose convolution computation here.
    // The code will involve launching threads over output dimensions
    // and computing the backward pass of a regular convolution.

    const int batch_size = input.size(0);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    const int output_h = output.size(2);
    const int output_w = output.size(3);

    CUDA_1D_KERNEL_LOOP(output_index, batch_size * out_channels * output_h * output_w) {
        int w_out = output_index % output_w;
        int h_out = (output_index / output_w) % output_h;
        int c_out = (output_index / (output_w * output_h)) % out_channels;
        int n = output_index / (out_channels * output_h * output_w);

        scalar_t val = 0;
        const int channel_group = c_out / groups;
        const int group_offset = c_out % groups;

        for (int k_h = 0; k_h < kernel_h; ++k_h) {
            for (int k_w = 0; k_w < kernel_w; ++k_w) {
                const int input_h_idx = (h_out + pad_h - dilation_h * k_h) / stride_h;
                const int input_w_idx = (w_out + pad_w - dilation_w * k_w) / stride_w;

                // Check if the current kernel position is valid in the input
                if ((h_out + pad_h - dilation_h * k_h) % stride_h != 0 ||
                    (w_out + pad_w - dilation_w * k_w) % stride_w != 0) {
                    continue;
                }

                if (input_h_idx < 0 || input_h_idx >= input_h ||
                    input_w_idx < 0 || input_w_idx >= input_w) {
                    continue;
                }

                for (int c_in = channel_group * (in_channels / groups);
                     c_in < (channel_group + 1) * (in_channels / groups);
                     ++c_in) {
                    const int weight_index = (c_out * kernel_h * kernel_w + k_h * kernel_w + k_w) * (in_channels / groups) + (c_in - channel_group * (in_channels / groups));
                    val += input[n][c_in][input_h_idx][input_w_idx] *
                           weight[c_out][c_in - channel_group*(in_channels/groups)][k_h][k_w];
                }
            }
        }
        output[n][c_out][h_out][w_out] = val;
    }
}

at::Tensor conv_transpose2d_forward_cuda(
    at::Tensor input,
    at::Tensor weight,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups) {

    auto output_height = (input.size(2) - 1) * stride_h - 2 * pad_h + 
                        dilation_h * (weight.size(2) - 1) + 1;
    auto output_width = (input.size(3) - 1) * stride_w - 2 * pad_w +
                        dilation_w * (weight.size(3) - 1) + 1;

    auto output = at::empty({input.size(0), weight.size(0), output_height, output_width}, input.options());

    const int batch_size = input.size(0);
    const int out_channels = weight.size(0);
    const int in_channels = input.size(1);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    dim3 blocks((batch_size * out_channels * output_height * output_width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_forward_cuda", ([&]{
        conv_transpose2d_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::current_stream()>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            out_channels, in_channels, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w,
            dilation_h, dilation_w, groups);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose2d_cpp_source = (
    "at::Tensor conv_transpose2d_forward_cuda(at::Tensor input, at::Tensor weight, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups);"
)

# Compile the custom kernel
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_forward_cuda"],
    verbose=True,
    extra_cflags=['-DTHON_CUDA'],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=(1, 1), padding=(0, 0), output_padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

        # Initialize weights similar to PyTorch's ConvTranspose2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) 

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Extract parameters
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dilation_h, dilation_w = self.dilation
        groups = self.groups

        # Call custom CUDA kernel
        out = conv_transpose2d.conv_transpose2d_forward_cuda(
            x, self.weight, stride_h, stride_w, 
            pad_h, pad_w, dilation_h, dilation_w, groups
        )

        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)

        return out