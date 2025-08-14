import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for 2D convolution
conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv2d_kernel(const scalar_t* __restrict__ input,
                             const scalar_t* __restrict__ weights,
                             scalar_t* __restrict__ output,
                             const int batch_size, const int in_channels,
                             const int out_channels, const int kernel_size,
                             const int input_height, const int input_width,
                             const int output_height, const int output_width,
                             const int stride, const int padding, const int dilation) {
    const int H_out = output_height;
    const int W_out = output_width;
    const int KW = kernel_size;
    const int KH = kernel_size;
    
    const int num_kernels = batch_size * out_channels * H_out * W_out;
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tx < num_kernels) {
        const int w_out = tx % W_out;
        const int h_out = (tx / W_out) % H_out;
        const int oc = (tx / (W_out * H_out)) % out_channels;
        const int batch_idx = tx / (out_channels * H_out * W_out);

        scalar_t val = 0;
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                const int h_in = h_out * stride - padding + kh * dilation;
                const int w_in = w_out * stride - padding + kw * dilation;
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        val += weights[oc * in_channels * KH * KW + ic * KH * KW + kh * KW + kw] *
                               input[batch_idx * in_channels * input_height * input_width +
                                     ic * input_height * input_width +
                                     h_in * input_width + w_in];
                    }
                }
            }
        }
        output[batch_idx * out_channels * H_out * W_out + oc * H_out * W_out + h_out * W_out + w_out] = val;
    }
}

std::tuple<torch::Tensor> conv2d_forward(torch::Tensor input, torch::Tensor weight,
                                          int stride, int padding, int dilation) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2); // Assuming square kernel
    
    const int KH = kernel_size;
    const int KW = kernel_size;
    
    // Output dimensions calculation
    const int output_height = (input_height + 2 * padding - dilation * (KH - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (KW - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    const int threads = 256;
    int blocks = (batch_size * out_channels * output_height * output_width + threads - 1) / threads;

    const int tensor_dim = input.dim();
    AT_ASSERT(tensor_dim == 4, "Input must be 4D tensor (batch, channels, height, width)");

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_forward", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, out_channels, kernel_size,
            input_height, input_width, output_height, output_width,
            stride, padding, dilation);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv2d_forward_cpp = """
std::tuple<torch::Tensor> conv2d_forward(torch::Tensor input, torch::Tensor weight,
                                         int stride, int padding, int dilation);
"""

conv2d = load_inline(
    name='conv2d',
    cpp_sources=conv2d_forward_cpp,
    cuda_sources=conv2d_source,
    functions=['conv2d_forward'],
    verbose=True,
    extra_cflags=['-DENABLE_CUDA'],
    extra_cuda_cflags=['-gencode=arch=compute_80,code=sm_80']  # Adjust based on GPU architecture
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias  # Note: This current implementation does not handle bias terms

        # Initialize weights similar to PyTorch's default
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Using He initialization

    def forward(self, x):
        return conv2d.conv2d_forward(x, self.weight, self.stride, self.padding, self.dilation)[0]