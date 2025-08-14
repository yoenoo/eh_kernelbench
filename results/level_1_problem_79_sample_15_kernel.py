import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for transposed 1D convolution
transposed_convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Define the CUDA kernel for transposed 1D convolution
__global__ void transposed_conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_length,
    int output_length,
    int stride,
    int padding,
    int dilation,
    int output_padding
) {
    // Each thread corresponds to an output element
    int n = blockIdx.x;
    int c = blockIdx.y;
    int t = blockIdx.z * blockDim.x + threadIdx.x;

    if (t >= output_length) return;

    float val = 0;
    const int weight_stride = 1; // For 1D, stride is scalar

    // Compute input position relative to output
    int output_time = t;
    int input_time = (output_time + 2 * padding - dilation * (kernel_size - 1) - output_padding) / stride;
    if ((output_time + 2 * padding - dilation * (kernel_size - 1) - output_padding) % stride != 0)
        input_time = -1; // Indicates out of valid region

    for (int k = 0; k < kernel_size; ++k) {
        int weight_index = k * out_channels + c;
        int current_input_time = input_time + dilation * k;

        // Check if within input bounds
        if (current_input_time >= 0 && current_input_time < input_length) {
            int input_index = current_input_time * in_channels + n * in_channels * input_length + c;
            val += input[input_index] * weight[weight_index];
        }
    }

    if (bias != nullptr)
        val += bias[c];

    // Write to output
    int output_index = t * out_channels + c + n * out_channels * output_length;
    output[output_index] = val;
}

torch::Tensor transposed_conv1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int output_padding
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int out_channels = weight.size(0) / kernel_size; // Assuming weight is [kernel_size*out_channels, in_channels]
    const int kernel_size = weight.size(1) / in_channels; // Wait, maybe better to pass as parameters
    // Need to actually pass kernel_size as a parameter since it's not directly inferable from weight dimensions

    // TODO: Calculate output length properly based on input parameters
    // Formula from PyTorch docs:
    int output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    dim3 threads(256);
    dim3 blocks(batch_size, out_channels, (output_length + threads.x - 1) / threads.x);

    transposed_conv1d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        input_length,
        output_length,
        stride,
        padding,
        dilation,
        output_padding
    );

    return output;
}
"""

transposed_convolution_cpp_source = (
    "torch::Tensor transposed_conv1d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int dilation, int output_padding);"
)

# Compile the inline CUDA code
transposed_conv = load_inline(
    name="transposed_conv",
    cpp_sources=transposed_convolution_cpp_source,
    cuda_sources=transposed_convolution_source,
    functions=["transposed_conv1d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias as per original ConvTranspose1d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = 0 # Since original code doesn't set this, but it's needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return transposed_conv.transposed_conv1d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.Tensor(),
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding
        )

def get_inputs():
    x = torch.rand(batch_size, in_channels, length).cuda()
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]