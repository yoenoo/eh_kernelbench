import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 1D convolution
conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose1d_kernel(scalar_t* __restrict__ output,
                                       const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ weight,
                                       int batch_size,
                                       int in_channels,
                                       int out_channels,
                                       int kernel_size,
                                       int input_length,
                                       int output_length,
                                       int stride,
                                       int padding) {
    const int B = blockIdx.x;
    const int C = blockIdx.y;
    const int K = threadIdx.x;
    
    const int out_channels_per_input = out_channels / in_channels;
    const int output_channel = C * out_channels_per_input + K % out_channels_per_input;
    const int input_channel = C;
    
    for (int o = K; o < output_length; o += blockDim.x) {
        scalar_t sum = 0;
        for (int k = 0; k < kernel_size; ++k) {
            const int i = (o - k - padding) / stride;
            if ((i >= 0) && (i < input_length) && ((o - k - padding) % stride == 0)) {
                sum += input[B * in_channels * input_length + input_channel * input_length + i] *
                       weight[output_channel * kernel_size + k];
            }
        }
        output[B * out_channels * output_length + output_channel * output_length + o] = sum;
    }
}

torch::Tensor conv_transpose1d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int kernel_size,
                                   int stride,
                                   int padding) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int input_length = input.size(2);
    
    int output_length = (input_length - 1) * stride + kernel_size - 2 * padding;
    
    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());
    
    const int threads = 256;
    dim3 blocks(batch_size, in_channels);
    dim3 threads_block(threads);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose1d_cuda", ([&] {
        conv_transpose1d_kernel<scalar_t><<<blocks, threads_block>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            input_length,
            output_length,
            stride,
            padding);
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose1d_cpp_source = (
    "torch::Tensor conv_transpose1d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding);"
)

# Compile the inline CUDA code for transposed 1D convolution
conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store convolution parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # Initialize weights similar to PyTorch's ConvTranspose1d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
            
        # Ensure groups is supported
        assert groups == 1, "Grouped convolutions not implemented in custom kernel"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv_transpose1d.conv_transpose1d_cuda(x.cuda(), self.weight.cuda(), self.kernel_size, self.stride, self.padding)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1).cuda()
        return output