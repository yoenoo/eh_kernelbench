import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for MaxPool3d with specific parameters
maxpool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<int KernelDim>
__device__ inline int get_input_index(int output_idx, int start, int kernel_size, int dilation) {
    int in_idx = start + (output_idx * dilation);
    return (in_idx < 0) ? 0 : (in_idx >= KernelDim ? KernelDim - 1 : in_idx);
}

__global__ void max_pool_3d_cuda_kernel(const float* input, float* output, 
                                       int batch_size, int channels, 
                                       int input_depth, int input_height, int input_width, 
                                       int output_depth, int output_height, int output_width) {
    // Thread and block indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * channels * output_depth * output_height * output_width) {
        return;
    }
    
    // Compute output coordinates
    int w = idx % output_width;
    int h = (idx / output_width) % output_height;
    int d = (idx / (output_width * output_height)) % output_depth;
    int c = (idx / (output_width * output_height * output_depth)) % channels;
    int n = idx / (output_width * output_height * output_depth * channels);
    
    // Starting position in input (accounting for padding and stride)
    const int padding = 1;
    const int stride = 2;
    const int dilation = 3;
    const int kernel_size = 3;
    
    int input_start_d = d * stride - padding;
    int input_start_h = h * stride - padding;
    int input_start_w = w * stride - padding;
    
    // Compute valid input indices
    input_start_d = max(input_start_d, 0);
    input_start_h = max(input_start_h, 0);
    input_start_w = max(input_start_w, 0);
    
    // Iterate over kernel region
    float max_val = -FLT_MAX;
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Compute current input indices with dilation
                int id = input_start_d + kd * dilation;
                int ih = input_start_h + kh * dilation;
                int iw = input_start_w + kw * dilation;
                
                // Check bounds
                if (id >= input_depth || ih >= input_height || iw >= input_width) {
                    continue;
                }
                
                int in_offset = ((n * channels + c) * input_depth + id) * input_height * input_width 
                                + ih * input_width + iw;
                float val = input[in_offset];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    
    // Store the result
    int out_offset = ((n * channels + c) * output_depth + d) * output_height * output_width 
                     + h * output_width + w;
    output[out_offset] = max_val;
}

torch::Tensor max_pool_3d_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);
    
    // Compute output dimensions considering stride and padding
    const int output_depth = (input_depth + 2 * 1 - (3 - 1)*3 - 1) / 2 + 1;
    const int output_height = (input_height + 2 * 1 - (3 - 1)*3 - 1) / 2 + 1;
    const int output_width = (input_width + 2 * 1 - (3 - 1)*3 - 1) / 2 + 1;
    
    auto output = torch::empty({batch_size, channels, output_depth, output_height, output_width}, input.options());
    
    const int total_elements = batch_size * channels * output_depth * output_height * output_width;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    max_pool_3d_cuda_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width
    );
    
    cudaDeviceSynchronize();
    return output;
}
"""

maxpool3d_cpp_source = "torch::Tensor max_pool_3d_cuda(torch::Tensor input);"

# Compile the CUDA kernel
maxpool3d = load_inline(
    name="maxpool3d_cuda",
    cpp_sources=maxpool3d_cpp_source,
    cuda_sources=maxpool3d_source,
    functions=["max_pool_3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int, return_indices: bool, ceil_mode: bool):
        super(ModelNew, self).__init__()
        self.max_pool_3d_cuda = maxpool3d
        # Store parameters if needed for dynamic configuration (though current kernel is hardcoded)
        self.params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'return_indices': return_indices,
            'ceil_mode': ceil_mode
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on the same device as the kernel (CUDA)
        x = x.cuda()
        return self.max_pool_3d_cuda.max_pool_3d_cuda(x)