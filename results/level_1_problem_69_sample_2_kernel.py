import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d with optimizations
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int in_channels, int out_channels, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w, int groups) {

    // Calculate output dimensions
    const int batch_size = input.size(0);
    const int out_h = output.size(2);
    const int out_w = output.size(3);
    const int in_h = input.size(2);
    const int in_w = input.size(3);

    // Thread indices
    const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;

    if (output_col >= out_w || output_row >= out_h) return;

    // Compute the input coordinates
    int in_row = (output_row - pad_h) / stride_h;
    int in_col = (output_col - pad_w) / stride_w;

    if ((output_row - pad_h) % stride_h != 0 || (output_col - pad_w) % stride_w != 0) return;

    // Channels are divided by groups
    const int group = 0; // Simplified for groups=1 case
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;

    CUDA_KERNEL_LOOP(output_idx, out_channels_per_group) {
        const int output_c = group * out_channels_per_group + output_idx;
        scalar_t val = 0;

        for (int kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {
            for (int kernel_col = 0; kernel_col < kernel_w; ++kernel_col) {
                int in_r = in_row + kernel_row * dilation_h;
                int in_c = in_col + kernel_col * dilation_w;

                if (in_r < 0 || in_r >= in_h || in_c <0 || in_c >= in_w) continue;

                for (int in_channel = 0; in_channel < in_channels_per_group; ++in_channel) {
                    val += weight[output_c][in_channel][kernel_row][kernel_col] *
                            input[batch][in_channel][in_r][in_c];
                }
            }
        }
        output[batch][output_c][output_row][output_col] = val;
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w, int groups) {

    // Output dimensions calculation (simplified for default values)
    auto input_size = input.sizes();
    int batch = input_size[0];
    int in_channels = input_size[1];
    int in_h = input_size[2];
    int in_w = input_size[3];

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int out_channels = weight.size(0);

    // Compute output spatial dimensions
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + 
        dilation_h * (kernel_h - 1) + 1 + pad_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + 
        dilation_w * (kernel_w - 1) + 1 + pad_w;

    auto output = torch::empty({batch, out_channels, out_h, out_w}, input.options());

    const int threads = 256;
    dim3 blocks((out_w + threads -1)/threads, (out_h + threads -1)/threads, batch);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            in_channels, out_channels, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w,
            dilation_h, dilation_w, groups);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w, int groups);
"""

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["--expt-extended-lambda"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=(1,1), padding=(0,0), output_padding=(0,0),
                 dilation=(1,1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

        # Initialize weight parameters
        kh, kw = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kh, kw))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

        self.cuda_conv = conv_transpose2d

    def forward(self, x):
        # Manually handle padding, stride, dilation, etc. since the kernel is simplified
        # Currently supports default output_padding and groups=1 case
        output = self.cuda_conv.conv_transpose2d_cuda(
            x, self.weight,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups
        )
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output