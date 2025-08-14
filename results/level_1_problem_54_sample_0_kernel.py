import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void conv3d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,5> input,
                                    const torch::PackedTensorAccessor<scalar_t,5> weight,
                                    torch::PackedTensorAccessor<scalar_t,5> output,
                                    const int batch_size, const int in_channels,
                                    const int depth, const int width, const int height,
                                    const int out_channels, const int kernel_size,
                                    const int stride, const int padding, const int dilation) {

    const int D_out = (depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int H_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int W_out = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    const int output_depth = blockIdx.z;
    const int output_height = blockIdx.y;
    const int output_width = blockIdx.x;
    const int channel_out = threadIdx.z;
    const int batch = threadIdx.y;
    const int channel_in = threadIdx.x;

    if (output_depth >= D_out || output_height >= H_out || output_width >= W_out ||
        channel_out >= out_channels || batch >= batch_size || channel_in >= in_channels) {
        return;
    }

    scalar_t sum = 0;
    for (int k = 0; k < kernel_size; ++k) {
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                const int d = output_depth * stride - padding + k * dilation;
                const int h = output_height * stride - padding + i * dilation;
                const int w = output_width * stride - padding + j * dilation;
                if (d < 0 || d >= depth || h < 0 || h >= height || w < 0 || w >= width) {
                    continue;
                }
                sum += input[batch][channel_in][d][h][w] * weight[channel_out][channel_in][k][i][j];
            }
        }
    }
    atomicAdd(&output[batch][channel_out][output_depth][output_height][output_width], sum);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv3d_forward(torch::Tensor input, torch::Tensor weight,
                                                                     int stride, int padding, int dilation) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto depth = input.size(2);
    const auto width = input.size(3);
    const auto height = input.size(4);

    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);

    auto output_depth = (depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    dim3 threads(1, 1, 1); // Channels, batches, out_channels
    dim3 blocks(output_depth, output_height, output_width);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_forward", ([&] {
        conv3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5>(),
            weight.packed_accessor<scalar_t,5>(),
            output.packed_accessor<scalar_t,5>(),
            batch_size, in_channels, depth, width, height,
            out_channels, kernel_size, stride, padding, dilation);
    }));

    cudaDeviceSynchronize();
    return std::make_tuple(output, input, weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_forward", &conv3d_forward, "Custom 3D convolution forward");
}
"""

conv3d_cpp_source = """
#include <torch/extension.h>
"""

# Compile the custom CUDA kernel
conv3d_module = load_inline(
    name="conv3d_kernel",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        # Only support groups=1 and bias=False for simplicity in this example
        assert groups == 1 and not bias, "Custom kernel currently supports only groups=1 and bias=False"
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        
    def forward(self, x):
        output, _, _ = conv3d_module.conv3d_forward(x, self.weight, self.stride, self.padding, self.dilation)
        return output