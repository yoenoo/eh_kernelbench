import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Transposed 1D Convolution
conv1d_transpose_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the CUDA kernel for ConvTranspose1D
__global__ void conv1d_transpose_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_length,
    int output_length,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    // Calculate the output position
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (output_idx >= batch_size * out_channels * output_length) {
        return;
    }
    
    int group = groups;
    int group_in_channels = in_channels / group;
    int group_out_channels = out_channels / group;

    // Compute indices
    int batch = output_idx / (out_channels * output_length);
    int out_ch = (output_idx / output_length) % out_channels;
    int out_pos = output_idx % output_length;
    
    // Find corresponding group
    int group_id = out_ch / group_out_channels;
    int out_ch_in_group = out_ch % group_out_channels;
    int in_ch_start = group_id * group_in_channels;
    
    for (int kernel_pos = 0; kernel_pos < kernel_size; kernel_pos++) {
        // Compute input position
        int input_pos = (out_pos + padding - kernel_pos * stride - output_padding);
        if (input_pos < 0 || input_pos >= input_length) {
            continue;
        }
        
        for (int in_ch = in_ch_start; in_ch < in_ch_start + group_in_channels; in_ch++) {
            int weight_idx = (out_ch_in_group * group_in_channels * kernel_size) + 
                            (in_ch - in_ch_start) * kernel_size + kernel_pos;
            int input_offset = batch * in_channels * input_length + in_ch * input_length + input_pos;
            int output_offset = output_idx;
            
            atomicAdd(&output[output_offset], input[input_offset] * weight[weight_idx]);
        }
    }
}

torch::Tensor conv1d_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    // Compute output length according to conv_transpose formula
    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());
    
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size * out_channels * output_length + threadsPerBlock - 1) / threadsPerBlock);
    
    conv1d_transpose_kernel<<<numBlocks, threadsPerBlock>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        input_length,
        output_length,
        stride,
        padding,
        output_padding,
        groups
    );
    
    cudaDeviceSynchronize();
    return output;
}
"""

conv1d_transpose_cpp_source = """
torch::Tensor conv1d_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding,
    int groups
);
"""

# Load the CUDA kernel
conv1d_transpose = load_inline(
    name="conv1d_transpose_cuda",
    cpp_sources=conv1d_transpose_cpp_source,
    cuda_sources=conv1d_transpose_source,
    functions=["conv1d_transpose_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Create the weight tensor for convolution
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Same as PyTorch's default
        
        # Bias handling (current version doesn't support bias, but can be added later)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
            
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize CUDA kernel as a member
        self.conv1d_transpose_op = conv1d_transpose

    def forward(self, x):
        # Perform convolution using custom CUDA kernel
        output = self.conv1d_transpose_op.conv1d_transpose_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )
        
        # Add bias if required
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
            
        return output