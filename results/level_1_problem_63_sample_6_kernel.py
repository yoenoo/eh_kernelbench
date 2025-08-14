import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv2D
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \\
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);    \\
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv2d_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input,
                             const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> weight,
                             torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output,
                             int kernel_size, int stride, int padding, int dilation) {
    int n = output.size(0);
    int c_out = output.size(1);
    int h_out = output.size(2);
    int w_out = output.size(3);

    CUDA_1D_KERNEL_LOOP(index, n * c_out * h_out * w_out) {
        int w_out_idx = index % w_out;
        int h_out_idx = (index / w_out) % h_out;
        int c_out_idx = (index / (w_out * h_out)) % c_out;
        int n_idx = index / (w_out * h_out * c_out);

        scalar_t val = 0;
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int h_in_idx = h_out_idx * stride + (i * dilation) - padding;
                int w_in_idx = w_out_idx * stride + (j * dilation) - padding;
                if (h_in_idx >= 0 && h_in_idx < input.size(2) && w_in_idx >= 0 && w_in_idx < input.size(3)) {
                    for (int c_in_idx = 0; c_in_idx < input.size(1); ++c_in_idx) {
                        val += input[n_idx][c_in_idx][h_in_idx][w_in_idx] * weight[c_out_idx][c_in_idx][i][j];
                    }
                }
            }
        }
        output[n_idx][c_out_idx][h_out_idx][w_out_idx] = val;
    }
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding, int dilation) {
    auto output_height = (input.size(2) + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto output_width = (input.size(3) + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto output = torch::zeros({input.size(0), weight.size(0), output_height, output_width}, input.options());

    dim3 threads(256);
    dim3 blocks((output.numel() + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_cuda", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            kernel_size, stride, padding, dilation
        );
    }));

    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding, int dilation);"
)

# Compile the inline CUDA code for Conv2D
conv2d_op = load_inline(
    name="conv2d_op",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        output = conv2d_op.conv2d_cuda(x, self.weight, self.kernel_size, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output