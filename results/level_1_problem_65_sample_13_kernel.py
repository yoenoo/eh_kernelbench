import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized ConvTranspose2d
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
      i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int out_channels, int in_channels, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int padding_h, int padding_w,
    int output_padding_h, int output_padding_w) {

    // Compute output coordinates
    int n = blockIdx.x;
    int c_out = blockIdx.y;
    int oh = threadIdx.y;
    int ow = threadIdx.x;

    scalar_t sum = 0;

    // Iterate over kernel elements and input channels
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            for (int c_in = 0; c_in < in_channels; ++c_in) {
                // Compute input coordinates
                int h_in = oh * stride_h - padding_h + kh;
                int w_in = ow * stride_w - padding_w + kw;

                // Check input boundaries
                if (h_in >= 0 && h_in < input.size(2) &&
                    w_in >= 0 && w_in < input.size(3)) {
                    // Compute weight index with reversed kernel positions
                    int w_pos = c_in * kernel_h * kernel_w + (kernel_h - 1 - kh) * kernel_w + (kernel_w - 1 - kw);
                    sum += input[n][c_in][oh][ow] * weight[c_in][c_out][kh][kw];
                }
            }
        }
    }

    output[n][c_out][oh][ow] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_h = input.size(2);
    const auto input_w = input.size(3);

    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    // Calculate output dimensions
    const auto output_h = (input_h - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    const auto output_w = (input_w - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;
    const auto out_channels = weight.size(0); // Output channels is the first dimension of weight

    auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());

    dim3 threads(input_w, input_h);
    dim3 blocks(batch_size * out_channels, 1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            out_channels, in_channels, kernel_h, kernel_w,
            stride_h, stride_w, padding_h, padding_w,
            output_padding_h, output_padding_w);
    }));

    return output;
}
"""

conv_transpose_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w, int output_padding_h, int output_padding_w);"
)

# Compile the inline CUDA code
conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weights like PyTorch's ConvTranspose2d
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
        # Extract parameters for the CUDA kernel
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        output_padding_h, output_padding_w = self.output_padding

        # Perform convolution with custom CUDA kernel
        output = conv_transpose.conv_transpose2d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            stride_h, stride_w,
            padding_h, padding_w,
            output_padding_h, output_padding_w
        )

        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output