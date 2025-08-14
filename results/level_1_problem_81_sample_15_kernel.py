import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for ConvTranspose2d
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
                                       const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
                                       torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
                                       int kernel_size,
                                       int stride,
                                       int padding,
                                       int dilation,
                                       int output_padding)
{
    const int b = blockIdx.z;
    const int y_out = blockIdx.y;
    const int x_out = blockIdx.x;
    const int channel_out = threadIdx.y;
    
    const int channels = input.size(1);
    const int kernel = kernel_size;
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_height = output.size(2);
    const int out_width = output.size(3);
    
    const int effective_kernel = (kernel - 1) * dilation + 1;
    
    int y_in = y_out * stride - padding;
    if (y_in < 0) y_in += output_padding;
    
    int x_in = x_out * stride - padding;
    if (x_in < 0) x_in += output_padding;
    
    scalar_t val = 0;
    
    for (int k = 0; k < kernel; ++k) {
        for (int l = 0; l < kernel; ++l) {
            const int y_k = y_in + k * dilation;
            const int x_l = x_in + l * dilation;
            
            if (y_k >= 0 && y_k < in_height && x_l >= 0 && x_l < in_width) {
                for (int c_in = 0; c_in < channels; ++c_in) {
                    const int weight_idx = channel_out * channels * kernel * kernel + c_in * kernel * kernel + k * kernel + l;
                    val += input[b][c_in][y_k][x_l] * weight[weight_idx][c_in][k][l];
                }
            }
        }
    }
    
    output[b][channel_out][y_out][x_out] = val;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding, int dilation, int output_padding) {
    const int batch = input.size(0);
    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);
    const int kernel = kernel_size;
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_h = (in_h - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_w = (in_w - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch, channels_out, out_h, out_w}, input.options());

    dim3 threads(1, channels_out, 1);
    dim3 blocks(in_w, in_h, batch);

    const int block_size = 256;
    const int num_blocks = (channels_out + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&]{
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            kernel_size, stride, padding, dilation, output_padding);
    }));

    return output;
}
"""

conv_transpose_cpp_source = "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding, int dilation, int output_padding);"

conv_transpose_mod = load_inline(
    name="conv_transpose_mod",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = 0  # Set based on input/output dimensions

        # Initialize weights similar to PyTorch's ConvTranspose2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        output = conv_transpose_mod.conv_transpose2d_cuda(
            x, 
            self.weight, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.output_padding
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output