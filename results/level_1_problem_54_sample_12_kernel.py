import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void custom_conv3d_forward(
    const float* input, const float* weight, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_width, int input_height,
    int kernel_size, int stride, int padding, int dilation,
    int output_depth, int output_width, int output_height) {

    const int batch_idx = blockIdx.x;
    const int out_channel = blockIdx.y;
    const int z = threadIdx.z;
    const int y = threadIdx.y;
    const int x = threadIdx.x;

    int output_z = z * stride - padding;
    int output_y = y * stride - padding;
    int output_x = x * stride - padding;

    float val = 0.0;
    for (int k = 0; k < in_channels; ++k) {
        for (int dz = 0; dz < kernel_size; ++dz) {
            for (int dy = 0; dy < kernel_size; ++dy) {
                for (int dx = 0; dx < kernel_size; ++dx) {
                    int iz = output_z + dz * dilation;
                    int iy = output_y + dy * dilation;
                    int ix = output_x + dx * dilation;
                    if (iz >= 0 && iz < input_depth && iy >= 0 && iy < input_width &&
                        ix >= 0 && ix < input_height) {
                        val += input[batch_idx * in_channels * input_depth * input_width * input_height +
                                    k * input_depth * input_width * input_height +
                                    dz * dilation * (input_width * input_height) +
                                    dy * dilation * input_height +
                                    dx * dilation] *
                               weight[out_channel * in_channels * kernel_size * kernel_size * kernel_size +
                                      k * kernel_size * kernel_size * kernel_size +
                                      dz * kernel_size * kernel_size +
                                      dy * kernel_size +
                                      dx];
                    }
                }
            }
        }
    }

    atomicAdd(&output[batch_idx * out_channels * output_depth * output_width * output_height +
                     out_channel * output_depth * output_width * output_height +
                     z * output_width * output_height +
                     y * output_height +
                     x], val);
}

torch::Tensor custom_conv3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_depth,
    int output_width,
    int output_height) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int input_depth = input.size(2);
    const int input_width = input.size(3);
    const int input_height = input.size(4);

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({batch_size, out_channels, output_depth, output_width, output_height}, output_options);

    dim3 threads(kernel_size, kernel_size, kernel_size);
    dim3 blocks(batch_size, out_channels);

    custom_conv3d_forward<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_depth, input_width, input_height,
        kernel_size, stride, padding, dilation,
        output_depth, output_width, output_height);

    cudaDeviceSynchronize();
    return output;
}
"""

conv3d_cpp_source = (
    "torch::Tensor custom_conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding, int dilation, int output_depth, int output_width, int output_height);"
)

# Compile the inline CUDA code for the custom 3D convolution
custom_conv3d = load_inline(
    name="custom_conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_kernel,
    functions=["custom_conv3d_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.custom_conv3d = custom_conv3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate output dimensions
        batch_size, in_channels, depth, width, height = x.shape
        output_depth = (depth + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        output_width = (width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        output_height = (height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        
        return self.custom_conv3d.custom_conv3d_forward_cuda(
            x,
            self.conv3d.weight,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            output_depth,
            output_width,
            output_height
        )