import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for transposed 1D convolution
conv_transpose_1d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel function for transposed 1D convolution
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_size,
    const int output_length,
    const int input_length,
    const int stride,
    const int padding,
    const int dilation) {

    const int batch_idx = blockIdx.x;
    const int out_channel = blockIdx.y;
    const int out_pos = blockIdx.z * blockDim.x + threadIdx.x;

    if (out_pos >= output_length) return;

    const int in_channel_offset = out_channel * in_channels;
    const int output_offset = (batch_idx * out_channels + out_channel) * output_length;

    for (int kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
        const int in_pos = (out_pos - padding - dilation * kernel_idx) / stride;
        if ((out_pos - padding - dilation * kernel_idx) % stride != 0 ||
            in_pos < 0 || in_pos >= input_length) {
            continue;  // Out of bounds due to padding/dilation/stride
        }

        const float weight_val = weight[in_channel_offset * kernel_size + kernel_idx];
        const float input_val = input[
            (batch_idx * in_channels + out_channel % in_channels) * input_length + in_pos];
        const int write_idx = output_offset + out_pos;
        atomicAdd(output + write_idx, weight_val * input_val);
    }
}

torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(1);

    // Compute output length
    const int output_length = (input_length - 1) * stride - 2 * padding + 
                            dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    const int block_size = 256;
    dim3 grid(batch_size, out_channels, 1);
    dim3 block(block_size);

    conv_transpose1d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        kernel_size, output_length, input_length,
        stride, padding, dilation);

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose_1d_cpp_source = (
    "torch::Tensor conv_transpose1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"
)

# Compile the CUDA kernel
conv_transpose_module = load_inline(
    name='conv_transpose',
    cpp_sources=conv_transpose_1d_cpp_source,
    cuda_sources=conv_transpose_1d_source,
    functions=['conv_transpose1d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        # Initialize weights similar to PyTorch's ConvTranspose1d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        output = conv_transpose_module.conv_transpose1d_cuda(
            x, self.weight, self.stride, self.padding, self.dilation
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        return output

def get_inputs():
    x = torch.rand(batch_size, in_channels, length).cuda()
    return [x.cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]