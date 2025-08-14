import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized Conv2D
conv2d_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void optimized_conv2d_kernel(const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ weight,
                                       scalar_t* output,
                                       const int batch_size,
                                       const int in_channels,
                                       const int out_channels,
                                       const int kernel_size,
                                       const int height,
                                       const int width,
                                       const int out_height,
                                       const int out_width,
                                       const int stride,
                                       const int padding,
                                       const int dilation) {
    const int output_spatial_size = out_height * out_width;
    const int channel_block_size = blockDim.z;
    const int channel_block_id = blockIdx.z;
    const int channel_id = channel_block_id * channel_block_size + threadIdx.z;
    
    // Calculate output spatial coordinates
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check boundaries
    if (w_out >= out_width || h_out >= out_height) return;

    scalar_t sum = 0;
    for (int c = 0; c < in_channels; c++) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Compute input coordinates with dilation and padding
                int h_in = h_out * stride + kh * dilation - padding;
                int w_in = w_out * stride + kw * dilation - padding;
                
                // Check input boundaries
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    scalar_t input_val = input[c * height * width + h_in * width + w_in];
                    scalar_t weight_val = weight[channel_id * in_channels * kernel_size * kernel_size +
                                                c * kernel_size * kernel_size + kh * kernel_size + kw];
                    sum += input_val * weight_val;
                }
            }
        }
    }
    
    // Write to output
    int output_index = channel_id * out_height * out_width + h_out * out_width + w_out;
    output[output_index] = sum;
}

std::tuple<torch::Tensor> optimized_conv2d_cuda(torch::Tensor input,
                                               torch::Tensor weight,
                                               int stride,
                                               int padding,
                                               int dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int height = input.size(2);
    const int width = input.size(3);
    
    // Compute output spatial dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    const dim3 threads(16, 16, 1); // X: width, Y: height, Z: out_channels per block
    const dim3 blocks(out_width / threads.x + 1,
                     out_height / threads.y + 1,
                     (out_channels + threads.z - 1) / threads.z);
    
    // Define the kernel launch configuration
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(input.type(), "optimized_conv2d_cuda", ([&] {
        optimized_conv2d_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            height,
            width,
            out_height,
            out_width,
            stride,
            padding,
            dilation);
    }));
    
    return std::tuple<torch::Tensor>(output);
}
"""

conv2d_kernel_header = """
#include <vector>
std::tuple<torch::Tensor> optimized_conv2d_cuda(torch::Tensor input,
                                               torch::Tensor weight,
                                               int stride,
                                               int padding,
                                               int dilation);
"""

# Compile the CUDA kernel
optimized_conv2d = load_inline(
    name="optimized_conv2d",
    cpp_sources=conv2d_kernel_header,
    cuda_sources=conv2d_kernel_source,
    functions="optimized_conv2d_cuda",
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        # Note: Bias not implemented in custom kernel for simplicity
        
        # Initialize weights similar to PyTorch's Conv2d
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        return optimized_conv2d.optimized_conv2d_cuda(
            x, self.weight, self.stride, self.padding, self.dilation)[0]