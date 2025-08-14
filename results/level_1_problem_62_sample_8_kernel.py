import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from torch.nn.parameter import Parameter

# Custom Conv2D CUDA implementation with optimizations
conv2d_custom_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv2d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,4> input,
                     const torch::PackedTensorAccessor<scalar_t,4> weight,
                     torch::PackedTensorAccessor<scalar_t,4> output,
                     int batch_size, int in_channels, int out_channels,
                     int kernel_h, int kernel_w,
                     int input_h, int input_w,
                     int output_h, int output_w,
                     int stride, int padding, int dilation) {

    const int H_out = output_h;
    const int W_out = output_w;

    const int num_kernels = batch_size * out_channels * H_out * W_out;
    CUDA_KERNEL_LOOP(index, num_kernels) {
        int w_out = index % W_out;
        int h_out = (index / W_out) % H_out;
        int c_out = (index / (W_out * H_out)) % out_channels;
        int n = index / (W_out * H_out * out_channels);

        scalar_t val = 0;
        for (int i = 0; i < in_channels; ++i) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int h_in = h_out * stride + kh * dilation - padding;
                    int w_in = w_out * stride + kw * dilation - padding;
                    if (h_in >= 0 && h_in < input_h && w_in >=0 && w_in < input_w) {
                        val += weight[c_out][i][kh][kw] * input[n][i][h_in][w_in];
                    }
                }
            }
        }
        output[n][c_out][h_out][w_out] = val;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_h = input.size(2);
    const auto input_w = input.size(3);

    const auto out_channels = weight.size(0);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    auto output_h = (input_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    auto output_w = (input_w + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());

    const int threads = 1024;
    int blocks = (batch_size * out_channels * output_h * output_w + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_forward", ([&] {
        using accscalar_t = torch::scalar_traits<scalar_t>::acc_type;
        
        conv2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            batch_size, in_channels, out_channels,
            kernel_h, kernel_w,
            input_h, input_w,
            output_h, output_w,
            stride, padding, dilation);
    }));

    return std::make_tuple(output, input, weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_forward", &conv2d_forward, "Custom 2D convolution forward");
}
"""

conv2d_cuda = load_inline(name="conv2d_cuda",
                         cpp_sources=[conv2d_custom_source],
                         extra_cuda_cflags=['-std=c++14', '-g', '--expt-extended-lambda'],
                         with_cuda=True,
                         verbose=True)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights like PyTorch Conv2D
        kernel_h, kernel_w = kernel_size
        self.weight = Parameter(torch.empty(out_channels, in_channels // groups, kernel_h, kernel_w))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Bias support can be added if needed
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            self.bias = None

    def forward(self, x):
        outputs, _, _ = conv2d_cuda.conv2d_forward(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            outputs += self.bias.view(1, -1, 1, 1)
        return outputs