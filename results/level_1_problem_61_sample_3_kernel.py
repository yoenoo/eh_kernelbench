import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Define kernel and bias (if any) manually since we are replacing the standard layer
        kernel_size_tuple = (kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size_tuple))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters (similar to PyTorch's default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Parameters for the custom convolution
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Load custom CUDA kernel
        self.custom_conv_t = load_inline(
            name='conv3d_transpose',
            cuda_sources='''
                #include <torch/extension.h>
                #include <ATen/cuda/CUDAContext.h>
                #include <stdio.h>

                #define CUDA_3D_KERNEL_LOOP(i, n)                        \
                for (int i = blockIdx.z * blockDim.z + threadIdx.z;      \
                     i < n;                                             \
                     i += blockDim.z * gridDim.z)

                template <typename scalar_t>
                __global__ void Conv3dTransposeKernel(
                    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
                    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> weight,
                    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
                    int kernel_depth, int kernel_height, int kernel_width,
                    int stride, int padding, int output_padding) {
                    
                    const int batch = blockIdx.x;
                    const int out_channel = blockIdx.y;
                    
                    CUDA_3D_KERNEL_LOOP(output_idx, output.size(2)*output.size(3)*output.size(4)) {
                        const int d_out = output_idx / (output.size(3)*output.size(4));
                        const int h_out = (output_idx % (output.size(3)*output.size(4))) / output.size(4);
                        const int w_out = output_idx % output.size(4);
                        
                        scalar_t val = 0;
                        for (int k_d = 0; k_d < kernel_depth; ++k_d) {
                            for (int k_h = 0; k_h < kernel_height; ++k_h) {
                                for (int k_w = 0; k_w < kernel_width; ++k_w) {
                                    const int d_in = (d_out - k_d - padding) / stride;
                                    if (d_in < 0 || d_in >= input.size(2))
                                        continue;
                                    const int h_in = (h_out - k_h - padding) / stride;
                                    if (h_in < 0 || h_in >= input.size(3))
                                        continue;
                                    const int w_in = (w_out - k_w - padding) / stride;
                                    if (w_in < 0 || w_in >= input.size(4))
                                        continue;
                                    // Assuming in_channels equals groups for simplicity here
                                    val += input[batch][out_channel][d_in][h_in][w_in] * 
                                            weight[out_channel][out_channel][k_d][k_h][k_w];
                                }
                            }
                        }
                        output[batch][out_channel][d_out][h_out][w_out] = val;
                    }
                }

                torch::Tensor conv3d_transpose_cuda(torch::Tensor input, torch::Tensor weight, 
                                                    int stride, int padding, int output_padding) {
                    auto output_options = torch::TensorOptions().like(input);
                    
                    // Compute output dimensions
                    int input_depth = input.size(2);
                    int input_height = input.size(3);
                    int input_width = input.size(4);
                    
                    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_depth + output_padding;
                    int output_height = (input_height - 1) * stride - 2 * padding + kernel_height + output_padding;
                    int output_width = (input_width - 1) * stride - 2 * padding + kernel_width + output_padding;
                    
                    auto output = torch::zeros({input.size(0), weight.size(0), output_depth, output_height, output_width}, output_options);
                    
                    const int threads = 256;
                    dim3 blocks(input.size(0), weight.size(0), 
                        ((output_depth * output_height * output_width) + threads - 1) / threads);
                    
                    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_transpose_cuda", ([&] {
                        Conv3dTransposeKernel<scalar_t><<<blocks, threads>>>(input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
                                                                           weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
                                                                           output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
                                                                           kernel_depth, kernel_height, kernel_width,
                                                                           stride, padding, output_padding);
                    }));
                    
                    return output;
                }
            ''',
            functions=['conv3d_transpose_cuda'],
            extra_cuda_cflags=['-lineinfo'],
            verbose=True
        )

    def forward(self, x):
        return self.custom_conv_t.conv3d_transpose_cuda(
            x, 
            self.weight, 
            self.stride, 
            self.padding, 
            self.output_padding
        )