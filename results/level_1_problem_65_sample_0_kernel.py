import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/macros/Macros.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int batch_size, int in_channels, int out_channels,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int padding_h, int padding_w, int output_padding_h, int output_padding_w) {

    CUDA_1D_KERNEL_LOOP(index, batch_size * out_channels * output.size(2) * output.size(3)) {
        int out_col = index % output.size(3);
        int out_row = (index / output.size(3)) % output.size(2);
        int oc = (index / (output.size(2) * output.size(3))) % out_channels;
        int n = index / (out_channels * output.size(2) * output.size(3));

        int in_row = (out_row + padding_h - output_padding_h) / stride_h;
        int in_col = (out_col + padding_w - output_padding_w) / stride_w;

        if ((out_row + padding_h - output_padding_h) % stride_h != 0 ||
            (out_col + padding_w - output_padding_w) % stride_w != 0) {
            continue;
        }

        in_row = (out_row + padding_h - output_padding_h) / stride_h;
        in_col = (out_col + padding_w - output_padding_w) / stride_w;

        scalar_t val = 0;
        for (int kc = 0; kc < in_channels; ++kc) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int h_idx = in_row + kh;
                    int w_idx = in_col + kw;
                    if (h_idx < 0 || h_idx >= input.size(2) || w_idx < 0 || w_idx >= input.size(3)) {
                        continue;
                    }
                    val += weight[kc][oc][kh][kw] * input[n][kc][h_idx][w_idx];
                }
            }
        }
        output[n][oc][out_row][out_col] = val;
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                                   int padding_h, int padding_w, int output_padding_h, int output_padding_w) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(1); // since weight is [in_channels, out_channels, kernel_h, kernel_w]

    // Compute output dimensions
    auto H_out = (input.size(2) - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    auto W_out = (input.size(3) - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::empty({batch_size, out_channels, H_out, W_out}, input.options());

    dim3 blocks儋(
        (batch_size * out_channels * H_out * W_out + 1024 - 1) / 1024
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, in_channels, out_channels,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, output_padding_h, output_padding_w);
    }));

    return output;
}
"""

conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, "
    "int kernel_h, int kernel_w, int stride_h, int stride_w, "
    "int padding_h, int padding_w, int output_padding_h, int output_padding_w);"
)

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weights similar to ConvTranspose2d but transposed
        self.weight = nn.Parameter(torch.empty((in_channels, out_channels, *kernel_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        self.conv_transpose_cuda = conv_transpose2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv_transpose_cuda.conv_transpose2d_cuda(
            x.cuda(),
            self.weight.cuda(),
            self.kernel_size[0], self.kernel_size[1],
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.output_padding[0], self.output_padding[1]
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output