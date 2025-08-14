import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA convolution kernel implementation
convolution2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void conv2d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
                                     torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weights,
                                     torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
                                     const int batch_size, const int in_channels, const int out_channels,
                                     const int kernel_h, const int kernel_w,
                                     const int input_h, const int input_w,
                                     const int output_h, const int output_w,
                                     const int stride, const int padding_h, const int padding_w) {
    
    const int H_out = output_h;
    const int W_out = output_w;
    
    const int num_kernels = batch_size * out_channels * H_out * W_out;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_kernels) {
        const int w_out = idx % W_out;
        const int h_out = (idx / W_out) % H_out;
        const int b = (idx / (H_out * W_out)) % batch_size;
        const int c_out = idx / (batch_size * H_out * W_out);
        
        scalar_t sum = 0;
        for (int k_h = 0; k_h < kernel_h; ++k_h) {
            for (int k_w = 0; k_w < kernel_w; ++k_w) {
                const int h_in = h_out * stride - padding_h + k_h;
                const int w_in = w_out * stride - padding_w + k_w;
                
                if (h_in >= 0 && h_in < input_h && w_in >=0 && w_in < input_w) {
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        sum += input[b][c_in][h_in][w_in] * weights[c_out][c_in][k_h][k_w];
                    }
                }
            }
        }
        output[b][c_out][h_out][w_out] = sum;
    }
}

at::Tensor conv2d_forward_cuda(const at::Tensor& input, const at::Tensor& weights, 
                              int stride, int padding_h, int padding_w) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    
    const int kernel_h = weights.size(2);
    const int kernel_w = weights.size(3);
    
    const int out_channels = weights.size(0);
    
    const int output_h = (input_h + 2 * padding_h - (kernel_h - 1) - 1) / stride + 1;
    const int output_w = (input_w + 2 * padding_w - (kernel_w - 1) - 1) / stride + 1;
    
    auto output = at::zeros({batch_size, out_channels, output_h, output_w}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * output_h * output_w + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_forward_cuda", ([&] {
        conv2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weights.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, in_channels, out_channels,
            kernel_h, kernel_w,
            input_h, input_w,
            output_h, output_w,
            stride, padding_h, padding_w
        );
    }));
    
    return output;
}
"""

# Inline CUDA kernel definition for the module
convolution2d_cpp_source = """
at::Tensor conv2d_forward_cuda(const at::Tensor& input, const at::Tensor& weights, 
                              int stride, int padding_h, int padding_w);
"""

# Load the CUDA extension
convolution2d_module = load_inline(
    name="convolution2d",
    cpp_sources=convolution2d_cpp_source,
    cuda_sources=convolution2d_source,
    functions=["conv2d_forward_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Currently only supports stride=1 and padding=0 for simplicity
        assert stride == 1 and padding == 0, "Only stride=1 and padding=0 supported"
        assert dilation == 1 and groups == 1 and not bias, "Need to add support for these options"
        
        self.stride = stride
        self.padding_h = kernel_size[0] // 2  # For symmetric padding
        self.padding_w = kernel_size[1] // 2
        
        # Initialize weights as a learnable parameter
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        
        # Bias support would require adding another parameter and kernel modification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return convolution2d_module.conv2d_forward_cuda(
            x, self.weight, self.stride, self.padding_h, self.padding_w
        )