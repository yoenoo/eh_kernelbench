import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename scalar_t>
__global__ void conv2d_kernel(const scalar_t* __restrict__ input, 
                             const scalar_t* __restrict__ weight, 
                             scalar_t* __restrict__ output,
                             int batch_size, int in_channels, int out_channels,
                             int input_height, int input_width, 
                             int kernel_size, 
                             int output_height, int output_width) {

    // Each thread computes one output element
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = blockIdx.z;

    if (w >= output_width || h >= output_height || c_out >= out_channels) 
        return;

    scalar_t sum = 0;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h * 1 + kh; // Assuming stride=1 and padding=0 for simplicity
                int w_in = w * 1 + kw;
                if (h_in < input_height && w_in < input_width) {
                    sum += input[c_in * input_height * input_width + h_in * input_width + w_in] *
                           weight[c_out * in_channels * kernel_size * kernel_size + 
                                  c_in * kernel_size * kernel_size + 
                                  kh * kernel_size + kw];
                }
            }
        }
    }
    output[c_out * output_height * output_width + h * output_width + w] = sum;
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int output_height = input_height - kernel_size + 1;
    const int output_width = input_width - kernel_size + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    const dim3 threads(16, 16);
    dim3 blocks((output_width + threads.x - 1)/threads.x, 
                (output_height + threads.y - 1)/threads.y, 
                out_channels);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_cuda", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), 
            weight.data_ptr<scalar_t>(), 
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_size, 
            output_height, output_width);
    }));

    return output;
}
"""

cpp_source = "torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight);"

# Compile the inline CUDA code
conv2d_op = load_inline(
    name="conv2d_op",
    cpp_sources=cpp_source,
    cuda_sources=convolution_source,
    functions=["conv2d_cuda"],
    verbose=True,
    extra_cflags=["-DGETConvolution_BACKWARD0"],
    extra_cuda_cflags=["-gencode=arch=compute_80,code=sm_80"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights similar to PyTorch's Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        # Register the custom CUDA op
        self.conv2d_op = conv2d_op

    def forward(self, x):
        # Currently implemented for stride=1, padding=0 for simplicity
        # Custom implementation can be extended for other parameters
        out = self.conv2d_op.conv2d_cuda(x, self.weight)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out