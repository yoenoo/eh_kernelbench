import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom ConvTranspose2d implementation using a fused CUDA kernel
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>

template <typename T>
__global__ void conv_transpose2d_kernel(const T* input, const T* weight, T* output,
                                       int batch_size, int in_channels, int out_channels,
                                       int input_height, int input_width, int output_height, int output_width,
                                       int kernel_h, int kernel_w, int stride_h, int stride_w,
                                       int padding_h, int padding_w) {

    const int output_channels = out_channels;
    const int kernel_size = kernel_h * kernel_w;
    
    int batch_idx = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = blockIdx.z;

    T val = static_cast<T>(0);
    int input_row = out_y / stride_h;
    int in_offset_y = out_y - input_row * stride_h;
    
    int input_col = out_x / stride_w;
    int in_offset_x = out_x - input_col * stride_w;

    if (input_row < input_height && input_col < input_width) {

        for (int k_h = 0; k_h < kernel_h; ++k_h) {
            int kh = kernel_h - 1 - k_h;
            int input_row_idx = input_row + kh - padding_h;
            if (input_row_idx < 0 || input_row_idx >= input_height) continue;
            
            for (int k_w = 0; k_w < kernel_w; ++k_w) {
                int kw = kernel_w - 1 - k_w;
                int input_col_idx = input_col + kw - padding_w;
                if (input_col_idx < 0 || input_col_idx >= input_width) continue;
                
                for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                    const T w = weight[(in_ch * kernel_h * kernel_w + kh * kernel_w + kw) * output_channels + batch_idx * output_channels + ?]; // Need to check weight storage layout
                    const T in_val = input[batch_idx * in_channels * input_height * input_width + in_ch * input_height * input_width + input_row_idx * input_width + input_col_idx];
                    val += w * in_val;
                }
            }
        }
        output[batch_idx * output_channels * output_height * output_width + out_ch * output_height * output_width + out_y * output_width + out_x] = val;
    }
}

at::Tensor conv_transpose2d_cuda(at::Tensor input, at::Tensor weight,
                                int stride_h, int stride_w,
                                int padding_h, int padding_w) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int output_height = input.size(2) * stride_h - 2 * padding_h;
    const int output_width = input.size(3) * stride_w - 2 * padding_w;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels = weight.size(0); // Assuming weight is [out_channels, in_channels, kh, kw]

    auto output = at::empty({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 blocks(batch_size, output_height, output_width);
    dim3 threads(1024);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(), weight.data<scalar_t>(), output.data<scalar_t>(),
            batch_size, in_channels, out_channels,
            input.size(2), input.size(3), output_height, output_width,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w);
    }));

    return output;
}
"""

# Inline CUDA compilation
conv_transpose2d_cpp = """
at::Tensor conv_transpose2d_cuda(
    at::Tensor input,
    at::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w
);
"""
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        output = conv_transpose2d.conv_transpose2d_cuda(
            x,
            self.weight,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1]
        )
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
            
        return output