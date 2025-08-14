import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D convolution
conv1d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv1d_kernel(const scalar_t* __restrict__ x, const scalar_t* __restrict__ weight, scalar_t* __restrict__ y, int batch_size, int in_channels, int out_channels, int length, int kernel_size, int stride, int padding, int dilation) {
    // Calculate input and output dimensions
    int output_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    CUDA_KERNEL_LOOP(index, batch_size * out_channels * output_length) {
        int batch_idx = index / (out_channels * output_length);
        int out_channel = (index / output_length) % out_channels;
        int out_pos = index % output_length;

        // Compute the start and end of the input region
        int in_start = out_pos * stride - padding;
        int in_end = in_start + dilation * (kernel_size - 1) + 1;

        // Ensure input indices are within bounds
        in_start = fmax(in_start, 0);
        in_end = fmin(in_end, length);

        // Calculate the effective kernel start and end
        int kernel_start = fmax((padding - out_pos * stride) / dilation, 0);
        int kernel_end = kernel_size - ((out_pos * stride + padding) + dilation * (kernel_size - 1) - (length - 1) + 1 > 0 ? (out_pos * stride + padding) + dilation * (kernel_size - 1) - (length - 1) + 1 : 0);

        scalar_t val = 0;
        for (int kernel_idx = kernel_start; kernel_idx < kernel_end; kernel_idx++) {
            int in_pos = out_pos * stride + kernel_idx * dilation - padding;
            if (in_pos < 0 || in_pos >= length) continue;
            for (int group_idx = 0; group_idx < in_channels / out_channels; group_idx++) {  // Assuming groups divide evenly
                int in_channel = group_idx + out_channel * (in_channels / out_channels);
                val += x[batch_idx * in_channels * length + (in_channel * length + in_pos)] *
                       weight[out_channel * in_channels / out_channels * kernel_size + (group_idx * kernel_size + kernel_idx)];
            }
        }
        y[index] = val;
    }
}

torch::Tensor conv1d_cuda(torch::Tensor x, torch::Tensor weight, int stride, int padding, int dilation, int groups) {
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto out_channels = weight.size(0);
    auto length = x.size(2);
    auto kernel_size = weight.size(2);

    int output_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto y = torch::zeros({batch_size, out_channels, output_length}, x.options());

    dim3 threads(256);
    dim3 blocks((batch_size * out_channels * output_length + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "conv1d_cuda", ([&] {
        conv1d_kernel<scalar_t><<<blocks, threads>>>(
            x.data<scalar_t>(), weight.data<scalar_t>(),
            y.data<scalar_t>(), batch_size, in_channels, out_channels,
            length, kernel_size, stride, padding, dilation
        );
    }));

    return y;
}
"""

conv1d_cpp_source = """
torch::Tensor conv1d_cuda(torch::Tensor x, torch::Tensor weight, int stride, int padding, int dilation, int groups);
"""

# Compile the inline CUDA code for convolution
conv1d = load_inline(
    name="conv1d",
    cuda_sources=conv1d_source,
    cpp_sources=conv1d_cpp_source,
    functions=["conv1d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        self.conv1d_op = conv1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1d_op.conv1d_cuda(x, self.weight, self.stride, self.padding, self.dilation, self.groups)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1)
        return y