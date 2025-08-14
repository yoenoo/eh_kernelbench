import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose_2d_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
                                       const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
                                       torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
                                       int kernel_h, int kernel_w,
                                       int stride_h, int stride_w,
                                       int padding_h, int padding_w,
                                       int output_padding_h, int output_padding_w,
                                       int groups,
                                       int batch, int in_channels, int out_channels_per_group,
                                       int input_h, int input_w,
                                       int output_h, int output_w) {
    CUDA_1D_KERNEL_LOOP(index, batch * output_h * output_w * out_channels_per_group) {
        int out_c = (index / (output_h * output_w)) % out_channels_per_group;
        int out_y = (index / output_w) % output_h;
        int out_x = index % output_w;
        int batch_idx = index / (out_channels_per_group * output_h * output_w);

        int in_c_group = out_c; // assuming groups
        int in_c = in_c_group + (groups > 1 ? (batch_idx % groups) * out_channels_per_group : 0);

        scalar_t val = 0;
        for (int k_h = 0; k_h < kernel_h; ++k_h) {
            int in_y = (out_y + padding_h) - k_h * stride_h - output_padding_h;
            if (in_y < 0 || in_y >= input_h) continue;
            for (int k_w = 0; k_w < kernel_w; ++k_w) {
                int in_x = (out_x + padding_w) - k_w * stride_w - output_padding_w;
                if (in_x < 0 || in_x >= input_w) continue;
                for (int g = 0; g < groups; ++g) {
                    int weight_offset = (g * out_channels_per_group + out_c) * kernel_h * kernel_w * in_channels;
                    val += input[batch_idx][in_c + g * (in_channels/groups)][in_y][in_x] *
                            weight[g * out_channels_per_group + out_c][in_c_group][k_h][k_w];
                }
            }
        }
        output[batch_idx][out_c][out_y][out_x] = val;
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int kernel_h, int kernel_w,
                                   int stride_h, int stride_w,
                                   int padding_h, int padding_w,
                                   int output_padding_h, int output_padding_w,
                                   int groups) {
    auto batch = input.size(0);
    auto in_channels = input.size(1);
    auto input_h = input.size(2);
    auto input_w = input.size(3);

    auto out_channels = weight.size(0) * groups; // depends on group configuration
    auto out_channels_per_group = weight.size(1); // out_channels / groups

    // Compute output dimensions
    int output_h = (input_h - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int output_w = (input_w - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::empty({batch, out_channels, output_h, output_w}, input.options());

    dim3 blocks(TORCH_GET_BLOCKS(output.numel()));
    dim3 threads(TORCH_GET_THREADS());

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            output_padding_h, output_padding_w,
            groups,
            batch, in_channels, out_channels_per_group,
            input_h, input_w,
            output_h, output_w);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int kernel_h, int kernel_w,
                                   int stride_h, int stride_w,
                                   int padding_h, int padding_w,
                                   int output_padding_h, int output_padding_w,
                                   int groups);
"""

conv_transpose_2d = load_inline(
    name="conv_transpose_2d",
    cpp_sources=cpp_source,
    cuda_sources=conv_transpose_2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.groups = groups
        
        # Initialize weights similar to ConvTranspose2d but transposed
        kernel_h, kernel_w = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_h, kernel_w))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.conv_transpose_2d = conv_transpose_2d

    def forward(self, x):
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        output_padding_h, output_padding_w = self.output_padding

        out = self.conv_transpose_2d.conv_transpose2d_cuda(
            x, self.weight, 
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            output_padding_h, output_padding_w,
            self.groups)
        
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
            
        return out