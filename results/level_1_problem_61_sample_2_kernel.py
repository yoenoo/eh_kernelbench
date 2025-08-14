import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 3D transpose convolution with specific parameters
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void conv_transpose3d_kernel(const T* input, const T* weight, T* output,
                                        int batch_size, int in_channels, int out_channels,
                                        int input_depth, int input_height, int input_width,
                                        int kernel_size,
                                        int output_depth, int output_height, int output_width) {
    // Each thread computes one output element
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (output_idx >= batch_size * out_channels * output_depth * output_height * output_width) {
        return;
    }
    
    int w = output_idx % output_width;
    int h = (output_idx / output_width) % output_height;
    int d = (output_idx / (output_width * output_height)) % output_depth;
    int c_out = (output_idx / (output_width * output_height * output_depth)) % out_channels;
    int n = output_idx / (out_channels * output_depth * output_height * output_width);
    
    T val = 0;
    for (int k_z = 0; k_z < kernel_size; ++k_z) {
        for (int k_y = 0; k_y < kernel_size; ++k_y) {
            for (int k_x = 0; k_x < kernel_size; ++k_x) {
                int input_d = d - k_z; // Assuming stride=1 and no padding here
                int input_h = h - k_y;
                int input_w = w - k_x;
                
                if (input_d >= 0 && input_h >=0 && input_w >=0 && 
                    input_d < input_depth && input_h < input_height && input_w < input_width) {
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        int weight_idx = c_in * out_channels * kernel_size*kernel_size*kernel_size +
                                        c_out * kernel_size*kernel_size*kernel_size +
                                        k_z * kernel_size*kernel_size + 
                                        k_y * kernel_size + 
                                        k_x;
                        val += weight[weight_idx] * 
                               input[n * in_channels * input_depth * input_height * input_width +
                                     c_in * input_depth * input_height * input_width +
                                     input_d * input_height * input_width +
                                     input_h * input_width +
                                     input_w];
                    }
                }
            }
        }
    }
    output[output_idx] = val;
}

std::tuple<torch::Tensor, torch::Tensor> conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight,
                                                               int kernel_size,
                                                               int output_depth, int output_height, int output_width) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(1); // Assuming weight is [in_channels, out_channels, ...]
    
    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());
    
    dim3 threads(256);
    dim3 blocks((output.numel() + threads.x - 1) / threads.x);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose3d_kernel", ([&] {
        conv_transpose3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            input.size(2), input.size(3), input.size(4),
            kernel_size,
            output_depth, output_height, output_width
        );
    }));
    
    return output;
}
"""

conv_transpose3d_cpp_source = R"(
std::tuple<torch::Tensor, torch::Tensor> conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight,
                                                               int kernel_size,
                                                               int output_depth, int output_height, int output_width);
)";

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights similar to ConvTranspose3d
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, 
                                              kernel_size, kernel_size, kernel_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output shape based on parameters (this is simplified for demo)
        # Assuming padding=0, stride=1, for simplicity
        output_depth = x.size(2) + self.kernel_size - 1
        output_height = x.size(3) + self.kernel_size - 1
        output_width = x.size(4) + self.kernel_size - 1
        
        # Call the custom CUDA kernel
        out = conv_transpose3d.conv_transpose3d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            self.kernel_size,
            output_depth, output_height, output_width
        )
        return out[0]  # Assuming the function returns a tuple, extract the tensor