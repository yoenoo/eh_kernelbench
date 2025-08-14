import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input,
                                       const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> weight,
                                       torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output,
                                       int in_channels, int out_channels, int kernel_size,
                                       int stride, int padding, int dilation) {
    int batch = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = blockIdx.z;
    
    scalar_t sum = 0;
    for (int k_chan = 0; k_chan < in_channels; k_chan++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int dilated_ky = ky * dilation;
                int dilated_kx = kx * dilation;
                
                int in_y = out_y + stride - dilated_ky - 1 - padding;
                int in_x = out_x + stride - dilated_kx - 1 - padding;
                
                if (in_y >= 0 && in_y < input.size(2) && in_x >= 0 && in_x < input.size(3)) {
                    int weight_idx = k_chan * kernel_size * kernel_size + ky * kernel_size + kx;
                    int input_idx = batch * in_channels * input.size(2) * input.size(3) + k_chan * input.size(2) * input.size(3) + in_y * input.size(3) + in_x;
                    int output_idx = batch * out_channels * output.size(2) * output.size(3) + (out_channels - 1) * output.size(2) * output.size(3) + out_y * output.size(3) + out_x;
                    
                    sum += input[batch][k_chan][in_y][in_x] * weight[k_chan][out_channels - 1][ky][kx];
                }
            }
        }
    }
    output[batch][out_channels - 1][out_y][out_x] = sum;
}

std::vector<torch::Tensor> conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, 
                                                int stride, int padding, int dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    
    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({batch_size, out_channels, out_height, out_width}, output_options);
    
    dim3 threads(1);
    dim3 blocks(batch_size, out_height, out_width);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            in_channels, out_channels, kernel_size, stride, padding, dilation);
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose2d_cpp_source = """
std::vector<torch::Tensor> conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, 
                                                int stride, int padding, int dilation);
"""

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights similar to PyTorch's default initialization
        self.weight = nn.Parameter(torch.empty((in_channels, out_channels, kernel_size, kernel_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Call custom CUDA kernel
        output = conv_transpose2d.conv_transpose2d_cuda(
            x, self.weight, self.stride, self.padding, self.dilation)[0]
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        
        return output