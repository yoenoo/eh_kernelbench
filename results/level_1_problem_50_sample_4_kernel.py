import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel implementation for 2D convolution
conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void custom_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size, int input_channels, int input_height, int input_width,
    int output_channels, int kernel_size, int stride, int padding,
    int output_height, int output_width
) {
    // Each thread computes one output element
    const int batch_id = blockIdx.x;
    const int out_y = blockIdx.y;
    const int out_x = blockIdx.z;
    const int out_channel = threadIdx.x;

    if (out_channel >= output_channels) return;

    // Compute input coordinates
    const int in_y = out_y * stride - padding;
    const int in_x = out_x * stride - padding;

    scalar_t sum = 0.0;
    // Iterate over kernel and input channels
    for (int k_y = 0; k_y < kernel_size; ++k_y) {
        const int y = in_y + k_y;
        if (y < 0 || y >= input_height) continue;
        for (int k_x = 0; k_x < kernel_size; ++k_x) {
            const int x = in_x + k_x;
            if (x < 0 || x >= input_width) continue;
            for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
                const int weight_idx = out_channel * input_channels * kernel_size * kernel_size +
                                      in_channel * kernel_size * kernel_size +
                                      k_y * kernel_size + k_x;
                const int input_idx = batch_id * input_channels * input_height * input_width +
                                      in_channel * input_height * input_width +
                                      y * input_width + x;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    // Store result in output
    const int output_idx = batch_id * output_channels * output_height * output_width +
                          out_channel * output_height * output_width +
                          out_y * output_width + out_x;
    output[output_idx] = sum;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> custom_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding
) {
    const int batch_size = input.size(0);
    const int input_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_channels = weight.size(0);
    const int kernel_size = weight.size(2); // Assuming square kernel
    const int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    torch::Tensor output = torch::empty({batch_size, output_channels, output_height, output_width}, input.options());
    
    dim3 threads(output_channels); // Each thread handles one output channel
    dim3 blocks(batch_size, output_height, output_width);

    // Determine the kernel function based on input type
    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv2d", ([&]{
        custom_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, input_channels, input_height, input_width,
            output_channels, kernel_size, stride, padding,
            output_height, output_width
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

# The header and declarations required for compilation
conv2d_cpp_header = "std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> custom_conv2d(torch::Tensor input, torch::Tensor weight, int stride, int padding);"

# Compile the CUDA code
conv2d = load_inline(
    name='custom_conv2d',
    cpp_sources=conv2d_cpp_header,
    cuda_sources=conv2d_source,
    functions=['custom_conv2d'],
    verbose=True,
    extra_cflags=['-std=c++14'],
    extra_cuda_cflags=['-std=c++14']
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        # Store parameters of the original conv layer to replicate
        self.in_channels = 3
        self.out_channels = 96
        self.kernel_size = 11
        self.stride = 4
        self.padding = 2
        
        # Initialize weights similar to PyTorch's default
        self.weight = nn.Parameter(torch.empty(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        # The custom CUDA function requires the parameters
        output = conv2d.custom_conv2d(x, self.weight, self.stride, self.padding)[0]
        return output

# Note: The above code may need additional adjustments such as proper error checking,
#       handling of bias (if any), and ensuring gradients are computed (backprop implementation).