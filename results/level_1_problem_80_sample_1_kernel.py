import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv2d_kernel(const scalar_t* input, const scalar_t* weight, scalar_t* output,
                             int batch_size, int in_channels, int out_channels,
                             int input_height, int input_width,
                             int kernel_h, int kernel_w,
                             int pad_h, int pad_w,
                             int stride, int dilation_h, int dilation_w,
                             int output_height, int output_width) {
    
    CUDA_1D_KERNEL_LOOP(output_idx, batch_size * out_channels * output_height * output_width) {
        int w_out = output_idx % output_width;
        int h_out = (output_idx / output_width) % output_height;
        int c_out = (output_idx / (output_width * output_height)) % out_channels;
        int n = output_idx / (out_channels * output_height * output_width);

        scalar_t sum = 0;
        for (int k_h = 0; k_h < kernel_h; ++k_h) {
            for (int k_w = 0; k_w < kernel_w; ++k_w) {
                int h_in = h_out * stride + k_h * dilation_h - pad_h;
                int w_in = w_out * stride + k_w * dilation_w - pad_w;
                
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        sum += weight[c_out * in_channels * kernel_h * kernel_w + c_in * kernel_h * kernel_w + k_h * kernel_w + k_w] *
                               input[n * in_channels * input_height * input_width + c_in * input_height * input_width +
                                     h_in * input_width + w_in];
                    }
                }
            }
        }
        output[output_idx] = sum;
    }
}

std::tuple<int, int> compute_output_size(int input_height, int input_width,
                                        int kernel_h, int kernel_w,
                                        int stride, int pad_h, int pad_w,
                                        int dilation_h, int dilation_w) {
    int output_h = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int output_w = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;
    return std::make_tuple(output_h, output_w);
}

torch::Tensor custom_conv2d(torch::Tensor input, torch::Tensor weight,
                           int kernel_h, int kernel_w,
                           int stride, int pad_h, int pad_w,
                           int dilation_h, int dilation_w) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    auto output_dims = compute_output_size(input_height, input_width,
                                          kernel_h, kernel_w,
                                          stride, pad_h, pad_w,
                                          dilation_h, dilation_w);
    int output_height = std::get<0>(output_dims);
    int output_width = std::get<1>(output_dims);

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 blocks((output_height * output_width * out_channels + 512 - 1) / 512);
    dim3 threads(512);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_kernel", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride, dilation_h, dilation_w,
            output_height, output_width);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

convolution_cpp_source = """
std::tuple<int, int> compute_output_size(int, int,
                                        int, int,
                                        int, int, int,
                                        int, int);
torch::Tensor custom_conv2d(torch::Tensor input, torch::Tensor weight,
                           int kernel_h, int kernel_w,
                           int stride, int pad_h, int pad_w,
                           int dilation_h, int dilation_w);
"""

conv_ops = load_inline(
    name="conv_ops",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["custom_conv2d"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=(0,0), dilation=(1,1), bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights (weights need to be loaded from original model)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 
                                              kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        output = conv_ops.custom_conv2d(
            x,
            self.weight.view(self.out_channels, -1, 1),  # Reshape weight for kernel
            self.kernel_size[0], self.kernel_size[1],
            self.stride, self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output