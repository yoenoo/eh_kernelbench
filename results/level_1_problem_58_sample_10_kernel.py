import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D ConvTranspose operation
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> input,
                                const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> weight,
                                torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> output,
                                const int batch_size,
                                const int in_channels,
                                const int out_channels,
                                const int kernel_d,
                                const int kernel_h,
                                const int kernel_w,
                                const int stride_d,
                                const int stride_h,
                                const int stride_w,
                                const int padding_d,
                                const int padding_h,
                                const int padding_w,
                                const int output_padding_d,
                                const int output_padding_h,
                                const int output_padding_w,
                                const int groups) {

    const int d_out = blockIdx.z;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y / gridDim.y;

    if (h_out >= output.size(3) || w_out >= output.size(4)) {
        return;
    }

    scalar_t val = 0;

    const int in_d_offset = (d_out + output_padding_d) - padding_d;
    const int in_h_offset = (h_out + output_padding_h) - padding_h;
    const int in_w_offset = (w_out + output_padding_w) - padding_w;

    const int d_in_start = in_d_offset / stride_d;
    const int h_in_start = in_h_offset / stride_h;
    const int w_in_start = in_w_offset / stride_w;

    const int kernel_d_start = (in_d_offset >= 0) ? 0 : ((-in_d_offset + stride_d - 1) / stride_d);
    const int kernel_h_start = (in_h_offset >= 0) ? 0 : ((-in_h_offset + stride_h - 1) / stride_h);
    const int kernel_w_start = (in_w_offset >= 0) ? 0 : ((-in_w_offset + stride_w - 1) / stride_w);

    const int kernel_d_end = std::min(kernel_d_start + ((-in_d_offset) / stride_d +1), kernel_d);
    const int kernel_h_end = std::min(kernel_h_start + ((-in_h_offset)/stride_h +1), kernel_h);
    const int kernel_w_end = std::min(kernel_w_start + ((-in_w_offset)/stride_w +1), kernel_w);

    const int d_in_start_clamped = std::max(d_in_start, 0);
    const int h_in_start_clamped = std::max(h_in_start, 0);
    const int w_in_start_clamped = std::max(w_in_start, 0);

    const int d_in_end = (d_in_start_clamped + kernel_d_start * stride_d) > input.size(2) ?
        input.size(2) : (d_in_start_clamped + kernel_d_end * stride_d);
    const int h_in_end = (h_in_start_clamped + kernel_h_start * stride_h) > input.size(3) ?
        input.size(3) : (h_in_start_clamped + kernel_h_end * stride_h);
    const int w_in_end = (w_in_start_clamped + kernel_w_start * stride_w) > input.size(4) ?
        input.size(4) : (w_in_start_clamped + kernel_w_end * stride_w);

    for (int d_k = kernel_d_start; d_k < kernel_d_end; ++d_k) {
        const int d_in = d_in_start_clamped + d_k * stride_d;
        if (d_in >= input.size(2)) {
            continue;
        }

        for (int h_k = kernel_h_start; h_k < kernel_h_end; ++h_k) {
            const int h_in = h_in_start_clamped + h_k * stride_h;
            if (h_in >= input.size(3)) {
                continue;
            }

            for (int w_k = kernel_w_start; w_k < kernel_w_end; ++w_k) {
                const int w_in = w_in_start_clamped + w_k * stride_w;
                if (w_in >= input.size(4)) {
                    continue;
                }

                const auto in_val = input[batch][d_in][h_in][w_in];
                const auto weight_val = weight[d_out][d_k][h_k][w_k];

                val += in_val * weight_val;
            }
        }
    }

    output[batch][d_out][h_out][w_out] = val;
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight,
                                int kernel_d, int kernel_h, int kernel_w,
                                int stride_d, int stride_h, int stride_w,
                                int padding_d, int padding_h, int padding_w,
                                int output_padding_d, int output_padding_h, int output_padding_w,
                                int groups) {
    // Output dimensions calculation
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto depth_out = (input.size(2) - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    auto height_out = (input.size(3) - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    auto width_out = (input.size(4) - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    dim3 threads(16, 16); // Thread block size
    dim3 blocks(div_up(width_out, threads.x), div_up(height_out, threads.y), depth_out);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose3d_cuda", ([&] {
        conv_transpose3d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            batch_size, in_channels, out_channels, kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w, padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w, groups);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = """
#include <torch/extension.h>

#define div_up(a, b) ((a) + (b) - 1) / (b)

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight,
                                int kernel_d, int kernel_h, int kernel_w,
                                int stride_d, int stride_h, int stride_w,
                                int padding_d, int padding_h, int padding_w,
                                int output_padding_d, int output_padding_h, int output_padding_w,
                                int groups);
"""

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=cpp_source,
    cuda_sources=conv_transpose_3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=["-DForceCPU"],
    extra_cuda_cflags=["-Wno-deprecated-gpu-targets"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 output_padding: tuple = (0, 0, 0), groups: int = 1,
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        # Initialize weight similar to PyTorch's ConvTranspose3d
        kernel_d, kernel_h, kernel_w = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_d, kernel_h, kernel_w))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unpack parameters
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        output_padding_d, output_padding_h, output_padding_w = self.output_padding

        # Call CUDA kernel
        output = conv_transpose3d.conv_transpose3d_cuda(
            x, self.weight, kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w,
            self.groups
        )

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)

        return output