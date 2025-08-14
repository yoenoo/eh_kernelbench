import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized Conv3D with kernel_depth=1
conv3d_custom_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 16

template <typename scalar_t>
__global__ void conv3d_kernel(const scalar_t* __restrict__ input,
                             const scalar_t* __restrict__ weight,
                             scalar_t* __restrict__ output,
                             const int batch_size,
                             const int in_channels,
                             const int out_channels,
                             const int kernel_h,
                             const int kernel_w,
                             const int stride_h,
                             const int stride_w,
                             const int pad_h,
                             const int pad_w,
                             const int input_h,
                             const int input_w,
                             const int input_d,
                             const int output_h,
                             const int output_w,
                             const int output_d) {

    const int n = blockIdx.z;
    const int out_channel = blockIdx.y;
    const int out_y = blockIdx.x * blockDim.y + threadIdx.y;
    const int out_x = threadIdx.x;

    if (out_y >= output_h || out_x >= output_w) return;

    scalar_t sum = 0;

    for (int d = 0; d < input_d; d++) {
        const int in_d = d;
        const int out_d = d; // since kernel depth is 1, output depth = input depth

        for (int k_y = 0; k_y < kernel_h; ++k_y) {
            for (int k_x = 0; k_x < kernel_w; ++k_x) {
                int in_y = out_y * stride_h - pad_h + k_y;
                int in_x = out_x * stride_w - pad_w + k_x;
                
                // Check input boundaries
                if (in_y < 0 || in_y >= input_h || in_x < 0 || in_x >= input_w) continue;
                
                for (int c = 0; c < in_channels; ++c) {
                    scalar_t w_val = weight[out_channel * in_channels * kernel_h * kernel_w + 
                                            c * kernel_h * kernel_w + 
                                            k_y * kernel_w + k_x];
                    scalar_t in_val = input[n * in_channels * input_h * input_w * input_d + 
                                           c * input_h * input_w * input_d + 
                                           in_y * input_w * input_d + 
                                           in_x * input_d + d];
                    sum += w_val * in_val;
                }
            }
        }
    }

    const int out_idx = n * out_channels * output_h * output_w * output_d + 
                       out_channel * output_h * output_w * output_d + 
                       out_y * output_w * output_d + 
                       out_x * output_d + 
                       out_d;

    output[out_idx] = sum;
}

at::Tensor conv3d_forward(const at::Tensor input,
                         const at::Tensor weight,
                         int stride_h,
                         int stride_w,
                         int pad_h,
                         int pad_w) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    const int input_d = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int kernel_d = weight.size(4); // which should be 1 as per model design

    const int output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    const int output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    const int output_d = input_d; // since kernel depth is 1 and no stride/padding in depth
    
    at::Tensor output = at::zeros({batch_size, out_channels, output_h, output_w, output_d}, input.type());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((output_w + threads.x - 1) / threads.x, (output_h + threads.y - 1) / threads.y, batch_size * out_channels);

    const int kernel_h_val = kernel_h;
    const int kernel_w_val = kernel_w;
    const int stride_h_val = stride_h;
    const int stride_w_val = stride_w;
    const int pad_h_val = pad_h;
    const int pad_w_val = pad_w;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_forward", ([&]{
        conv3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, out_channels,
            kernel_h_val, kernel_w_val,
            stride_h_val, stride_w_val,
            pad_h_val, pad_w_val,
            input_h, input_w, input_d,
            output_h, output_w, output_d);
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = """
at::Tensor conv3d_forward(const at::Tensor input,
                         const at::Tensor weight,
                         int stride_h,
                         int stride_w,
                         int pad_h,
                         int pad_w);
"""

conv3d_custom = load_inline(
    name="conv3d_custom",
    cpp_sources=cpp_source,
    cuda_sources=conv3d_custom_source,
    functions=["conv3d_forward"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = (stride, stride, 1)
        self.padding = (padding, padding, 0)
        self.dilation = (dilation, dilation, 1)
        self.groups = groups
        self.bias = bias
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, 1))
        self.bias_param = nn.Parameter(torch.empty(out_channels)) if bias else None
        
        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)
    
    def forward(self, x):
        return conv3d_custom.conv3d_forward(
            x, 
            self.weight,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1]
        )