import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Conv2d
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;    \
       i += blockDim.x * gridDim.x)

using Tensor = torch::Tensor;

std::vector<int64_t> compute_output_size(Tensor input, int kernel_size, int stride, int padding, int dilation) {
    int input_size = input.size(3); // assuming width dimension
    int kernel_effective = dilation * (kernel_size - 1) + 1;
    int output_size = (input_size + 2 * padding - kernel_effective) / stride + 1;
    return {input.size(0), input.size(1), output_size}; // assuming square kernel and height same as width processing
}

// Note: This is a simplified example and may not handle all edge cases or general cases like PyTorch's conv2d
// For a real scenario, implement a proper convolution kernel with loops over channels, output dimensions, etc.
__global__ void simple_conv2d_forward(const float* input, const float* weight, float* output,
                                     int batch, int in_channels, int out_channels,
                                     int input_width, int kernel_size, int output_width,
                                     int stride, int padding, int dilation) {

    CUDA_1D_KERNEL_LOOP(output_idx, batch * out_channels * output_width) {
        int output_col = output_idx % output_width;
        int output_row = (output_idx / output_width) % out_channels;
        int n = output_idx / (out_channels * output_width);

        float acc = 0;
        int kernel_radius = (kernel_size - 1) / 2; // assuming odd kernel size

        for (int f = 0; f < in_channels; ++f) {
            for (int k = 0; k < kernel_size; ++k) {
                int input_row = output_row * stride - padding + k * dilation;
                if (input_row < 0 || input_row >= input_width) continue;

                int input_col = output_col * stride - padding + k * dilation;
                if (input_col < 0 || input_col >= input_width) continue;

                int input_offset = n * in_channels * input_width * input_width +
                                   f * input_width * input_width +
                                   input_row * input_width + input_col;
                int weight_offset = output_row * in_channels * kernel_size * kernel_size +
                                    f * kernel_size * kernel_size + k * kernel_size + k; // simplified weight indexing
                acc += input[input_offset] * weight[weight_offset];
            }
        }
        output[output_idx] = acc;
    }
}

Tensor conv2d_forward_cuda(Tensor input, Tensor weight, int stride, int padding, int dilation) {
    const auto batch = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_height = input.size(2); // Note: This assumes height and width are same, might need adjustment
    const auto input_width = input.size(3);
    const auto kernel_size = weight.size(2); // assuming square kernel
    const auto out_channels = weight.size(0);

    auto output_size = compute_output_size(input, kernel_size, stride, padding, dilation);
    Tensor output = torch::zeros({batch, out_channels, output_size[2]}, input.options());

    int output_width = output_size[2];
    dim3 blocks((batch * out_channels * output_width + 1024 - 1) / 1024);
    dim3 threads(1024);

    simple_conv2d_forward<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        batch, in_channels, out_channels,
        input_width, kernel_size, output_width,
        stride, padding, dilation
    );

    cudaDeviceSynchronize();
    return output;
}
"""

conv2d_cpp_source = """
Tensor conv2d_forward_cuda(Tensor input, Tensor weight, int stride, int padding, int dilation);
"""

conv2d_op = load_inline(
    name="conv2d_op",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups  # Currently not used in the kernel
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels // groups, kernel_size, kernel_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        output = conv2d_op.conv2d_forward_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
        return output