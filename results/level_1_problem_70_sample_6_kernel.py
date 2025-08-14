import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 3D convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

at::Tensor conv_transpose3d_cuda(const at::Tensor& input, const at::Tensor& weight, 
                                const at::Tensor& bias, int64_t stride, 
                                int64_t padding, int64_t output_padding, 
                                int64_t dilation) {
    // Get tensor dimensions
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2); // Assuming cube kernel
    auto batch_size = input.size(0);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);

    // Compute output dimensions
    auto out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Initialize output tensor
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    // Grid and block dimensions
    const int threads_per_block = 256;
    const dim3 blocks(
        (batch_size * out_channels + threads_per_block - 1) / threads_per_block,
        out_depth,
        out_height * out_width
    );
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_cuda", ([&] {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        conv_transpose3d_kernel<scalar_t><<<blocks, threads_per_block, 0, stream>>>(
            input.contiguous().data<scalar_t>(),
            weight.contiguous().data<scalar_t>(),
            bias.contiguous().data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, out_channels,
            in_depth, in_height, in_width,
            kernel_size, stride, padding, dilation,
            output_padding
        );
    }));

    return output;
}

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int kernel_size, int stride, int padding, int dilation,
    int output_padding) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels) return;
    
    int b = idx / out_channels;
    int c_out = idx % out_channels;

    for (int d = blockIdx.y; d < in_depth; d += gridDim.y) {
        for (int h = blockIdx.z / out_width; h < in_height; h += gridDim.z / out_width) {
            for (int w = blockIdx.z % out_width; w < in_width; w += (gridDim.z / out_width)) {
                int in_offset = b * in_channels * in_depth * in_height * in_width
                            + c_in * in_depth * in_height * in_width
                            + d * in_height * in_width
                            + h * in_width
                            + w;
                
                for (int kd = 0; kd < kernel_size; ++kd) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int od = d * stride - padding + kd * dilation;
                            int oh = h * stride - padding + kh * dilation;
                            int ow = w * stride - padding + kw * dilation;
                            
                            // Apply output padding
                            if (od < 0 || od >= out_depth || oh < 0 || oh >= out_height || ow < 0 || ow >= out_width + output_padding) continue;
                            ow += output_padding; // Apply output padding to width dimension

                            int weight_offset = c_out * in_channels * kernel_size*kernel_size*kernel_size
                                            + c_in * kernel_size*kernel_size*kernel_size
                                            + kd * kernel_size*kernel_size
                                            + kh * kernel_size
                                            + kw;
                            
                            atomicAdd(&output[b * out_channels * out_depth * out_height * out_width
                                            + c_out * out_depth * out_height * out_width
                                            + od * out_height * out_width
                                            + oh * out_width
                                            + ow],
                                            input[in_offset] * weight[weight_offset]);
                        }
                    }
                }
            }
        }
    }
    if (bias) {
        atomicAdd(&output[idx], bias[c_out]);
    }
}
"""

# Compile the inline CUDA code
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=["-D_USE_CUDNN"],
    extra_cuda_cflags=["--expt-relaxed-constexpr"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights similar to PyTorch's ConvTranspose3d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 
                                               kernel_size, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose3d.conv_transpose3d_cuda(
            x, self.weight, self.bias if self.bias is not None else x.new_zeros(0),
            self.stride, self.padding, self.output_padding, self.dilation
        )