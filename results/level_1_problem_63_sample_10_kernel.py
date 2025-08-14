import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for Conv2D
conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Function to calculate output dimensions
static inline int output_size(int input_size, int kernel_size, int padding, int stride) {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

template <typename scalar_t>
__global__ void conv2d_kernel(const scalar_t* __restrict__ input,
                             const scalar_t* __restrict__ weight,
                             scalar_t* __restrict__ output,
                             const int batch_size,
                             const int in_channels,
                             const int out_channels,
                             const int kernel_size,
                             const int stride,
                             const int padding,
                             const int input_height,
                             const int input_width,
                             const int output_height,
                             const int output_width) {

    // Block and thread indices
    const int H = blockIdx.y;
    const int W = blockIdx.x;
    const int B = blockIdx.z / out_channels;
    const int C_out = blockIdx.z % out_channels;
    
    const int thread_x = threadIdx.x;
    const int thread_y = threadIdx.y;
    
    __shared__ scalar_t shared_input[32][32]; // Shared memory buffer for input patch
    
    scalar_t acc = 0.0;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            // Compute input indices
            int h_in = H * stride + kh - padding;
            int w_in = W * stride + kw - padding;
            
            if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                // Load input data into shared memory
                shared_input[thread_y][thread_x] = input[B * in_channels * input_height * input_width + 
                                                        (thread_y + h_in * in_channels) * input_width + 
                                                        (thread_x + w_in)];
                __syncthreads();
                
                // Multiply with weights and accumulate
                for (int c = 0; c < in_channels; ++c) {
                    acc += weight[C_out * in_channels * kernel_size * kernel_size + 
                                 c * kernel_size * kernel_size + 
                                 kh * kernel_size + kw] * 
                           shared_input[c][kh * kernel_size + kw];
                }
            }
        }
    }

    // Write result to output
    output[B * out_channels * output_height * output_width + 
           C_out * output_height * output_width + 
           H * output_width + W] = acc;
}

torch::Tensor conv2d_cuda(torch::Tensor input, 
                         torch::Tensor weight,
                         int kernel_size,
                         int stride,
                         int padding) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    const int output_height = output_size(input_height, kernel_size, padding, stride);
    const int output_width = output_size(input_width, kernel_size, padding, stride);

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, 
                              torch::dtype(input.scalar_type()).device(torch::kCUDA));

    dim3 threads(16, 16);
    dim3 blocks(output_width, output_height, batch_size * out_channels);

    const int blocks_per_sm = (gridDim.x * gridDim.y * gridDim.z) / 2048;
    const int shared_mem = threads.x * threads.y * sizeof(float);

    conv2d_kernel<float><<<blocks, threads, shared_mem>>>(input.data_ptr<float>(),
                                                         weight.data_ptr<float>(),
                                                         output.data_ptr<float>(),
                                                         batch_size,
                                                         in_channels,
                                                         out_channels,
                                                         kernel_size,
                                                         stride,
                                                         padding,
                                                         input_height,
                                                         input_width,
                                                         output_height,
                                                         output_width);
    
    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding);"
)

# Compile the inline CUDA code for Conv2D
conv2d = load_inline(
    name="conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=True,
    extra_cflags=["-g", "-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights similar to PyTorch's default initialization
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Ensure CUDA tensors
        x = x.cuda()
        weight = self.weight.cuda()
        
        # Call custom CUDA kernel
        output = conv2d.conv2d_cuda(x, weight, self.kernel_size, self.stride, self.padding)
        
        # Handle bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
            
        return output