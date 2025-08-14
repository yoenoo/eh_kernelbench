import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv2d_custom_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void custom_conv2d_kernel(const scalar_t* __restrict__ input,
                                    const scalar_t* __restrict__ weight,
                                    scalar_t* __restrict__ output,
                                    const int batch_size,
                                    const int in_channels,
                                    const int out_channels,
                                    const int kernel_h,
                                    const int kernel_w,
                                    const int input_h,
                                    const int input_w,
                                    const int output_h,
                                    const int output_w,
                                    const int stride_h,
                                    const int stride_w,
                                    const int padding_h,
                                    const int padding_w,
                                    const int dilation_h,
                                    const int dilation_w) {
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_channel = blockIdx.z;
    
    if (out_channel >= out_channels || h_out >= output_h || w_out >= output_w) {
        return;
    }
    
    scalar_t sum = 0.0;
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int input_h_idx = h_out * stride_h - padding_h + kh * dilation_h;
                int input_w_idx = w_out * stride_w - padding_w + kw * dilation_w;
                
                if (input_h_idx >= 0 && input_h_idx < input_h &&
                    input_w_idx >= 0 && input_w_idx < input_w) {
                    sum += weight[out_channel * in_channels * kernel_h * kernel_w + 
                                 in_ch * kernel_h * kernel_w + 
                                 kh * kernel_w + kw] *
                           input[in_ch * input_h * input_w +
                                 input_h_idx * input_w + input_w_idx];
                }
            }
        }
    }
    output[out_channel * output_h * output_w + h_out * output_w + w_out] = sum;
}

std::tuple<torch::Tensor> custom_conv2d_cuda(torch::Tensor input,
                                            torch::Tensor weight,
                                            int stride_h,
                                            int stride_w,
                                            int padding_h,
                                            int padding_w,
                                            int dilation_h,
                                            int dilation_w) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    
    const int out_channels = weight.size(0);
    
    const int output_h = (input_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int output_w = (input_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());
    
    const dim3 threads(32, 8);  // Block size 32 x 8 threads
    dim3 blocks(output_w, output_h, out_channels);
    
    const int shared_mem_size = 0;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "custom_conv2d_cuda", ([&] {
        custom_conv2d_kernel<scalar_t><<<blocks, threads, shared_mem_size, 
            torch::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            input_h,
            input_w,
            output_h,
            output_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w
        );
    }));
    
    return output;
}
"""

conv2d_cpp_source = (
    "std::tuple<torch::Tensor> custom_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);"
)

custom_conv2d = load_inline(
    name="custom_conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_custom_source,
    functions=["custom_conv2d_cuda"],
    verbose=False,
    extra_cflags=["-g", "-O3"],
    extra_cuda_cflags=["--use_fast_math"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Only supporting groups=1 and bias=False for simplicity in this example
        assert groups == 1 and not bias, "Only groups=1 and bias=False are supported"
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        self.reset_parameters()
        
        self.stride_h, self.stride_w = stride
        self.padding_h, self.padding_w = padding
        self.dilation_h, self.dilation_w = dilation
        
    def reset_parameters(self):
        # Initialize the weights (simplified; use proper initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        # Call the custom CUDA kernel
        output = custom_conv2d.custom_conv2d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w
        )[0]
        return output