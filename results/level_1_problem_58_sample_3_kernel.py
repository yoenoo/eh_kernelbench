import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized ConvTranspose3d
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Forward kernel for optimized transposed 3D convolution
template <typename scalar_t>
__global__ void conv_transpose3d_forward(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int out_depth, int out_height, int out_width,
    int in_depth, int in_height, int in_width
) {
    // Thread and block indices
    int n = blockIdx.z;
    int out_z = blockIdx.x * blockDim.z + threadIdx.z;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Calculate output index ranges
    int out_x = threadIdx.x;
    int output_offset = n * out_channels * out_depth * out_height * out_width +
                       out_z * out_height * out_width + 
                       out_y * out_width + out_x;
    
    scalar_t sum = 0;
    
    // Iterate over input channels and kernel weights
    for (int cin = 0; cin < in_channels; ++cin) {
        for (int kd = 0; kd < kD; ++kd) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    // Compute input position
                    int in_z = out_z + kd * strideD - padD;
                    int in_y = out_y + kh * strideH - padH;
                    int in_x = out_x + kw * strideW - padW;
                    
                    // Check valid input indices
                    if (in_z < 0 || in_z >= in_depth) continue;
                    if (in_y < 0 || in_y >= in_height) continue;
                    if (in_x < 0 || in_x >= in_width) continue;
                    
                    // Get weight index
                    int weight_offset = cin * out_channels * kD * kH * kW +
                                       kd * kH * kW * out_channels +
                                       kh * kW * out_channels +
                                       kw * out_channels;
                    
                    // Accumulate the product
                    for (int cout = 0; cout < out_channels; ++cout) {
                        sum += weight[weight_offset + cout] * 
                               input[n * in_channels * in_depth * in_height * in_width +
                                     cin * in_depth * in_height * in_width +
                                     in_z * in_height * in_width +
                                     in_y * in_width + in_x];
                    }
                }
            }
        }
    }
    
    // Store the result
    if (out_z < out_depth && out_y < out_height && out_x < out_width) {
        output[output_offset] = sum;
    }
}

// Wrapper function
at::Tensor conv_transpose3d_forward_cuda(
    at::Tensor input,
    at::Tensor weight,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW
) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_depth = input.size(2);
    const auto in_height = input.size(3);
    const auto in_width = input.size(4);
    
    const auto out_channels = weight.size(0); // Output channels is first dimension
    
    // Compute output size
    const auto out_depth = (in_depth - 1) * strideD - 2 * padD + kD;
    const auto out_height = (in_height - 1) * strideH - 2 * padH + kH;
    const auto out_width = (in_width - 1) * strideW - 2 * padW + kW;
    
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    const int threads = 256;
    dim3 blocks(
        (out_width + threads - 1) / threads,
        (out_height + threads - 1) / threads,
        batch_size
    );
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_forward", ([&] {
        conv_transpose3d_forward<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, out_channels,
            kD, kH, kW,
            strideD, strideH, strideW,
            padD, padH, padW,
            out_depth, out_height, out_width,
            in_depth, in_height, in_width
        );
    }));
    
    return output;
}
"""

# Compile the custom CUDA operator
conv_transpose3d = load_inline(
    name='conv_transpose3d',
    cpp_sources="""
        torch::Tensor conv_transpose3d_forward_cuda(
            torch::Tensor input,
            torch::Tensor weight,
            int kD, int kH, int kW,
            int strideD, int strideH, int strideW,
            int padD, int padH, int padW
        );
    """,
    cuda_sources=conv_transpose3d_source,
    functions=['conv_transpose3d_forward_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=(1, 1, 1), padding=(0, 0, 0),
                 output_padding=(0, 0, 0), groups=1, bias=False):
        super(ModelNew, self).__init__()
        
        # Register parameters and buffers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights similar to PyTorch's ConvTranspose3d
        weight_shape = (out_channels, in_channels // groups) + kernel_size
        self.weight = nn.Parameter(torch.empty(weight_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # Get kernel dimensions
        kD, kH, kW = self.kernel_size
        strideD, strideH, strideW = self.stride
        padD, padH, padW = self.padding
        
        # Compute output using custom CUDA operator
        output = conv_transpose3d.conv_transpose3d_forward_cuda(
            x.contiguous(), self.weight.contiguous(),
            kD, kH, kW,
            strideD, strideH, strideW,
            padD, padH, padW
        )
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        
        return output