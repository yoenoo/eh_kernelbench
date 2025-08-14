import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int batch_size, int in_channels, int out_channels,
    int kernel_size, int H_in, int W_in, int H_out, int W_out,
    int stride, int padding) {

    int h_out = blockIdx.x * blockDim.x + threadIdx.x;
    int w_out = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z;

    if (channel >= in_channels || h_out >= H_out || w_out >= W_out) return;

    scalar_t sum = 0;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            // Compute input coordinates
            int h_in = h_out * stride + kh - padding;
            int w_in = w_out * stride + kw - padding;
            // Check if in bounds
            if (h_in >= 0 && h_in < H_in && w_in >=0 && w_in < W_in) {
                sum += input[0][channel][h_in][w_in] * weight[channel][0][kh][kw];
            }
        }
    }

    output[0][channel][h_out][w_out] = sum;
}

torch::Tensor depthwise_conv2d_forward(torch::Tensor input, torch::Tensor weight, int stride, int padding) {
    // Get tensor dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0); // since groups = in_channels
    int kernel_size = weight.size(2);
    int H_in = input.size(2);
    int W_in = input.size(3);

    int H_out = (H_in + 2*padding - kernel_size) / stride + 1;
    int W_out = (W_in + 2*padding - kernel_size) / stride + 1;

    // Output tensor
    auto output = torch::zeros({batch_size, out_channels, H_out, W_out}, input.options());

    // Grid and block dimensions
    dim3 threads(16, 16); // Tune based on GPU
    dim3 blocks( (H_out + threads.x -1)/threads.x, (W_out + threads.y -1)/threads.y, in_channels);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "depthwise_conv2d_forward", ([&] {
        depthwise_conv2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, in_channels, out_channels,
            kernel_size, H_in, W_in, H_out, W_out,
            stride, padding);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = (
    "torch::Tensor depthwise_conv2d_forward(torch::Tensor input, torch::Tensor weight, int stride, int padding);"
)

# Compile the inline CUDA code
depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_forward"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))  # Depthwise weights
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        self.cuda_conv = depthwise_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cuda_conv.depthwise_conv2d_forward(x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out