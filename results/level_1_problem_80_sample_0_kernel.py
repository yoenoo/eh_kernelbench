import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized 2D convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

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
                                    const int stride,
                                    const int dilation_h,
                                    const int dilation_w,
                                    const int pad_h,
                                    const int pad_w) {

    const int output_size = output_h * output_w;
    const int num_kernels = kernel_h * kernel_w;
    const int channels_per_block = 32; // Channels per thread block

    const int block_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int block_ch = blockIdx.z * blockDim.z + threadIdx.z;

    // Calculate output spatial indices
    const int out_y = block_y / output_w;
    const int out_x = block_y % output_w;

    // Check if within output bounds
    if (block_x >= batch_size || out_y >= output_h || out_x >= output_w || block_ch >= out_channels) {
        return;
    }

    const int output_offset = (block_x * out_channels + block_ch) * output_h * output_w + out_y * output_w + out_x;
    
    scalar_t acc = 0;
    for (int k = 0; k < num_kernels; ++k) {
        const int kh = k / kernel_w;
        const int kw = k % kernel_w;
        
        // Compute dilated kernel positions
        const int dilated_kh = kh * dilation_h;
        const int dilated_kw = kw * dilation_w;
        
        // Compute input positions
        const int in_y = out_y * stride + dilated_kh - pad_h;
        const int in_x = out_x * stride + dilated_kw - pad_w;
        
        if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
            for (int c = 0; c < in_channels; c += channels_per_block) {
                const int channel_offset = c + block_ch % channels_per_block;
                if (channel_offset >= in_channels) break;
                
                acc += input[block_x * in_channels * input_h * input_w + 
                            (c + channel_offset) * input_h * input_w + 
                            in_y * input_w + in_x] *
                       weight[block_ch * in_channels * kernel_h * kernel_w + 
                              (c + channel_offset) * kernel_h * kernel_w + 
                              kh * kernel_w + kw];
            }
        }
    }

    output[output_offset] = acc;
}

at::Tensor custom_conv2d_cuda(at::Tensor input,
                             at::Tensor weight,
                             int stride,
                             int padding_h,
                             int padding_w,
                             int dilation_h,
                             int dilation_w,
                             int kernel_h,
                             int kernel_w) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    
    const int output_h = (input_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    const int output_w = (input_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;
    const int out_channels = weight.size(0);
    
    at::Tensor output = at::empty({batch_size, out_channels, output_h, output_w}, input.options());

    dim3 threads(1, 1, 1); // Block dimensions
    dim3 blocks(batch_size, output_h * output_w, out_channels); // Grid dimensions
    
    // Define block size based on available hardware
    threads.x = 1;
    threads.y = 8;
    threads.z = 8;

    const int shared_mem = 0;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    custom_conv2d_kernel<float><<<blocks, threads, shared_mem, stream>>>(
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
        stride,
        dilation_h,
        dilation_w,
        padding_h,
        padding_w
    );

    return output;
}
"""

convolution_cpp_source = """
at::Tensor custom_conv2d_cuda(
    at::Tensor input,
    at::Tensor weight,
    int stride,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int kernel_h,
    int kernel_w);
"""

# Compile the custom convolution kernel
custom_conv = load_inline(
    name='custom_conv',
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=['custom_conv2d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=(0,0), dilation=(1,1), bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
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
        output = custom_conv.custom_conv2d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.kernel_size[0],
            self.kernel_size[1]
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output