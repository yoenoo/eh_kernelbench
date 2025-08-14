import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized transposed convolution
conv_transpose_kernel = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

using torch::Tensor;

extern "C" __global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int input_channels,
    int output_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_height,
    int output_width) {

    CUDA_1D_KERNEL_LOOP(element_idx, batch_size * output_channels * output_height * output_width) {
        int w_idx = element_idx % output_width;
        int h_idx = (element_idx / output_width) % output_height;
        int oc = (element_idx / output_width / output_height) % output_channels;
        int n = element_idx / output_width / output_height / output_channels;

        float val = 0;
        for (int kc = 0; kc < input_channels; ++kc) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    // Compute the corresponding input coordinates
                    const int h_out = h_idx * stride - padding;
                    const int w_out = w_idx * stride - padding;
                    const int h_in = h_out - kh * dilation;
                    const int w_in = w_out - kw * dilation;

                    // Check if the input coordinates are within bounds
                    if (h_in >= 0 && w_in >= 0 && h_in < input_height && w_in < input_width) {
                        const int input_idx = n * input_channels * input_height * input_width +
                                             kc * input_height * input_width +
                                             h_in * input_width + w_in;

                        const int weight_idx = oc * input_channels * kernel_size * kernel_size +
                                             kc * kernel_size * kernel_size +
                                             kh * kernel_size + kw;

                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[element_idx] = val;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation) {

    const int batch_size = input.size(0);
    const int input_channels = input.size(1);
    const int output_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    // Compute output dimensions (simplified for illustration)
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + 2 * padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + 2 * padding;

    auto output = torch::zeros({batch_size, output_channels, output_height, output_width}, input.options());

    dim3 blocks(TORCH_DEFAULT_GRANULARITY);
    dim3 threads(256);
    int elements = batch_size * output_channels * output_height * output_width;
    int blocks_needed = (elements + threads.x - 1) / threads.x;
    if (blocks_needed > blocks.x) blocks.x = blocks_needed;

    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_channels,
        output_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        dilation,
        output_height,
        output_width);

    return output;
}
"""

conv_transpose_kernel_header = "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"

conv_transpose = load_inline(
    name="conv_transpose",
    cuda_sources=conv_transpose_kernel,
    cpp_sources=conv_transpose_kernel_header,
    functions="conv_transpose2d_cuda",
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        # Initialize weights (simple example, in practice use proper initialization)
        nn.init.xavier_normal_(self.weight)
        # Custom CUDA kernel from above
        self.conv_transpose_op = conv_transpose

    def forward(self, x):
        return self.conv_transpose_op.conv_transpose2d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.dilation
        )