import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized Conv2d
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \\
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// CUDA kernel for 2D convolution with dilation and padding
__global__ void custom_conv2d_kernel(
    const float* input, const float* weight, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_h, int kernel_w,
    int padding_h, int padding_w,
    int stride, int dilation_h, int dilation_w,
    int output_height, int output_width) {

    CUDA_1D_KERNEL_LOOP(output_idx, batch_size * out_channels * output_height * output_width) {
        int w_out = output_idx % output_width;
        int h_out = (output_idx / output_width) % output_height;
        int c_out = (output_idx / (output_width * output_height)) % out_channels;
        int n = output_idx / (out_channels * output_height * output_width);

        float acc = 0.0;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride + (kh * dilation_h) - padding_h;
                int w_in = w_out * stride + (kw * dilation_w) - padding_w;
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        acc += input[n * in_channels * input_height * input_width + c_in * input_height * input_width +
                                    h_in * input_width + w_in] *
                               weight[c_out * in_channels * kernel_h * kernel_w + c_in * kernel_h * kernel_w +
                                      kh * kernel_w + kw];
                    }
                }
            }
        }
        output[output_idx] = acc;
    }
}

torch::Tensor custom_conv2d(
    torch::Tensor input, torch::Tensor weight,
    int kernel_h, int kernel_w,
    int padding_h, int padding_w,
    int stride, int dilation_h, int dilation_w) {

    // Output dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);

    int output_height = (input_height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) / stride + 1;
    int output_width = (input_width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 blocks(256);
    dim3 threads((output_height * output_width * batch_size * out_channels + blocks.x - 1) / blocks.x);

    custom_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_h, kernel_w,
        padding_h, padding_w,
        stride, dilation_h, dilation_w,
        output_height, output_width);

    return output;
}
"""

cpp_source = """
torch::Tensor custom_conv2d(
    torch::Tensor input, torch::Tensor weight,
    int kernel_h, int kernel_w,
    int padding_h, int padding_w,
    int stride, int dilation_h, int dilation_w);
"""

custom_conv2d = load_inline(
    name="custom_conv2d",
    cpp_sources=cpp_source,
    cuda_sources=convolution_source,
    functions=["custom_conv2d"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=(0,0), dilation=(1,1), bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding_h, self.padding_w = padding
        self.dilation_h, self.dilation_w = dilation
        self.kernel_h, self.kernel_w = kernel_size
        # Initialize weights similar to PyTorch's Conv2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        # Bias is ignored as per original model's bias=False default

    def forward(self, x):
        return custom_conv2d(
            x, self.weight,
            self.kernel_h, self.kernel_w,
            self.padding_h, self.padding_w,
            self.stride, self.dilation_h, self.dilation_w
        )