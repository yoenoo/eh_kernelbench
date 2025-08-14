import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv3d_forward_kernel(const scalar_t* bottom_data, const scalar_t* weight_data, scalar_t* top_data,
    const int num, const int channels_out, const int channels_in,
    const int depth, const int height, const int width,
    const int kernel_size, const int stride, const int padding) {

    const int output_depth = (depth + 2 * padding - kernel_size) / stride + 1;
    const int output_height = (height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = width;

    CUDA_KERNEL_LOOP(index, num * channels_out * output_depth * output_height * output_width) {
        int n = index / (channels_out * output_depth * output_height * output_width);
        int c_out = (index / (output_depth * output_height * output_width)) % channels_out;
        int d = (index / (output_height * output_width)) % output_depth;
        int h = (index / output_width) % output_height;
        int w = index % output_width;

        scalar_t val = 0;
        for (int k = 0; k < kernel_size; ++k) {
            int d_in = d * stride - padding + k;
            if (d_in < 0 || d_in >= depth) continue;

            for (int kk = 0; kk < kernel_size; ++kk) {
                int h_in = h * stride - padding + kk;
                if (h_in < 0 || h_in >= height) continue;

                for (int s = 0; s < kernel_size; ++s) {
                    int w_in = w * stride + s;
                    if (w_in < 0 || w_in >= width) continue;

                    for (int c_in = 0; c_in < channels_in; ++c_in) {
                        const int bottom_offset =
                            n * channels_in * depth * height * width +
                            c_in * depth * height * width +
                            d_in * height * width +
                            h_in * width + w_in;
                        const int weight_offset =
                            c_out * channels_in * kernel_size * kernel_size * kernel_size +
                            c_in * kernel_size * kernel_size * kernel_size +
                            k * kernel_size * kernel_size +
                            kk * kernel_size + s;
                        val += bottom_data[bottom_offset] * weight_data[weight_offset];
                    }
                }
            }
        }

        top_data[index] = val;
    }
}

std::tuple<torch::Tensor, torch::Tensor> conv3d_forward(
        torch::Tensor input,
        torch::Tensor weight,
        int kernel_size,
        int stride,
        int padding) {

    const auto num = input.size(0);
    const auto channels_in = input.size(1);
    const auto depth = input.size(2);
    const auto height = input.size(3);
    const auto width = input.size(4);
    const auto channels_out = weight.size(0);

    auto output_depth = (depth + 2 * padding - kernel_size) / stride + 1;
    auto output_height = (height + 2 * padding - kernel_size) / stride + 1;
    auto output_width = width;

    auto output_size = torch::IntArrayRef({
        num,
        channels_out,
        output_depth,
        output_height,
        output_width
    });

    auto output = at::empty(output_size, input.options());
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_forward", ([&] {
        const scalar_t* bottom_data = input.data_ptr<scalar_t>();
        const scalar_t* weight_data = weight.data_ptr<scalar_t>();
        scalar_t* top_data = output.data_ptr<scalar_t>();

        dim3 threads(256);
        dim3 blocks((output.numel() + threads.x - 1) / threads.x);

        conv3d_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            bottom_data, weight_data, top_data,
            num, channels_out, channels_in,
            depth, height, width,
            kernel_size, stride, padding);
    }));

    return std::make_tuple(output, weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv3d_forward, "3D convolution forward");
}
"""

conv3d_cpp = load_inline(name='conv3d',
                        cuda_sources=conv3d_source,
                        extra_cuda_cflags=['-g','-O3'],
                        with_cuda=True)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, 
                 bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        self.kernel_size = kernel_size

    def forward(self, x):
        output, _ = conv3d_cpp.forward(x, self.weight, self.kernel_size, self.stride, self.padding)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        return output