import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 3D convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void ConvTranspose3DKernel(const scalar_t* input,
                                     const scalar_t* weight,
                                     scalar_t* output,
                                     int64_t batch_size,
                                     int64_t in_channels,
                                     int64_t out_channels,
                                     int64_t depth_out,
                                     int64_t height_out,
                                     int64_t width_out,
                                     int64_t kernel_depth,
                                     int64_t kernel_height,
                                     int64_t kernel_width,
                                     int64_t stride_depth,
                                     int64_t stride_height,
                                     int64_t stride_width,
                                     int64_t pad_depth,
                                     int64_t pad_height,
                                     int64_t pad_width,
                                     int64_t output_pad_depth,
                                     int64_t output_pad_height,
                                     int64_t output_pad_width) {

    int64_t batch_idx = blockIdx.x;
    int64_t out_d = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t out_h = blockIdx.z * blockDim.z + threadIdx.z;
    int64_t out_w = threadIdx.x;

    if (out_d >= depth_out || out_h >= height_out || out_w >= width_out) {
        return;
    }

    for (int64_t in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int64_t k_d = 0; k_d < kernel_depth; k_d++) {
            for (int64_t k_h = 0; k_h < kernel_height; k_h++) {
                for (int64_t k_w = 0; k_w < kernel_width; k_w++) {
                    // Calculate input indices
                    int64_t in_d = (out_d + pad_depth - k_d) / stride_depth;
                    int64_t in_h = (out_h + pad_height - k_h) / stride_height;
                    int64_t in_w = (out_w + pad_width - k_w) / stride_width;

                    // Check if within input bounds
                    if ((out_d + pad_depth - k_d) % stride_depth == 0 &&
                        in_d >= 0 && in_d < (depth_out + output_pad_depth) &&
                        (out_h + pad_height - k_h) % stride_height == 0 &&
                        in_h >= 0 && in_h < (height_out + output_pad_height) &&
                        (out_w + pad_width - k_w) % stride_width == 0 &&
                        in_w >= 0 && in_w < (width_out + output_pad_width)) {

                        const int64_t weight_idx = in_ch * kernel_depth * kernel_height * kernel_width * out_channels +
                                                   k_d * kernel_height * kernel_width * out_channels +
                                                   k_h * kernel_width * out_channels +
                                                   k_w * out_channels;

                        const int64_t input_offset = batch_idx * in_channels * (depth_out + output_pad_depth) * (height_out + output_pad_height) * (width_out + output_pad_width) +
                                                     in_ch * (depth_out + output_pad_depth) * (height_out + output_pad_height) * (width_out + output_pad_width) +
                                                     in_d * (height_out + output_pad_height) * (width_out + output_pad_width) +
                                                     in_h * (width_out + output_pad_width) + in_w;

                        const int64_t output_offset = batch_idx * out_channels * depth_out * height_out * width_out +
                                                      in_ch * depth_out * height_out * width_out +
                                                      out_d * height_out * width_out +
                                                      out_h * width_out + out_w;

                        atomicAdd(&output[output_offset], input[input_offset] * weight[weight_idx]);
                    }
                }
            }
        }
    }
}

std::vector<int64_t> output_size(int64_t batch_size,
                                int64_t in_depth, int64_t in_height, int64_t in_width,
                                int64_t kernel_depth, int64_t kernel_height, int64_t kernel_width,
                                int64_t stride_depth, int64_t stride_height, int64_t stride_width,
                                int64_t pad_depth, int64_t pad_height, int64_t pad_width,
                                int64_t output_pad_depth, int64_t output_pad_height, int64_t output_pad_width) {
    int64_t depth_out = (in_depth - 1) * stride_depth - 2 * pad_depth + kernel_depth + output_pad_depth;
    int64_t height_out = (in_height - 1) * stride_height - 2 * pad_height + kernel_height + output_pad_height;
    int64_t width_out = (in_width - 1) * stride_width - 2 * pad_width + kernel_width + output_pad_width;
    return {batch_size, in_channels, depth_out, height_out, width_out};
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int64_t stride_depth, int64_t stride_height, int64_t stride_width,
                                   int64_t pad_depth, int64_t pad_height, int64_t pad_width,
                                   int64_t output_pad_depth, int64_t output_pad_height, int64_t output_pad_width) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0); // weight size [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
    const auto kernel_depth = weight.size(2);
    const auto kernel_height = weight.size(3);
    const auto kernel_width = weight.size(4);

    const auto depth_in = input.size(2);
    const auto height_in = input.size(3);
    const auto width_in = input.size(4);

    auto out_sizes = output_size(batch_size, depth_in, height_in, width_in,
                                kernel_depth, kernel_height, kernel_width,
                                stride_depth, stride_height, stride_width,
                                pad_depth, pad_height, pad_width,
                                output_pad_depth, output_pad_height, output_pad_width);

    auto output = torch::zeros(out_sizes, input.options());

    const dim3 threads(32, 8, 8);
    dim3 blocks(batch_size, (out_sizes[2] + threads.y - 1)/threads.y, (out_sizes[3] + threads.z - 1)/threads.z);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_cuda", ([&] {
        ConvTranspose3DKernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            out_sizes[2], out_sizes[3], out_sizes[4],
            kernel_depth, kernel_height, kernel_width,
            stride_depth, stride_height, stride_width,
            pad_depth, pad_height, pad_width,
            output_pad_depth, output_pad_height, output_pad_width);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

# Compile the custom CUDA kernel
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias  # Note: Current implementation does not support bias

        # Initialize weights similar to PyTorch's ConvTranspose3d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure using CUDA tensors
        x = x.cuda()
        weight = self.weight.cuda()

        return conv_transpose3d.conv_transpose3d_cuda(
            x, weight,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2]
        )