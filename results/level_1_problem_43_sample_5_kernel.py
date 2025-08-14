import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

kernel_size = 3
stride = 2
padding = 1
dilation = 3

maxpool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void max_pool3d_forward(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t batch_size, int64_t channels,
    int64_t input_dim1, int64_t input_dim2, int64_t input_dim3,
    int64_t output_dim1, int64_t output_dim2, int64_t output_dim3,
    int kernel_size, int stride, int padding, int dilation) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;

    const int channel = blockIdx.w;
    const int batch = blockIdx.batch;

    if (idx >= output_dim1 || idy >= output_dim2 || idz >= output_dim3) return;

    int in_x = idx * stride - padding;
    int in_y = idy * stride - padding;
    int in_z = idz * stride - padding;

    scalar_t max_val = -INFINITY;
    
    // Iterate over the kernel volume
    for (int k = 0; k < kernel_size; k++) {
        int input_k = in_z + k * dilation;
        if (input_k < 0 || input_k >= input_dim3) continue;
        
        for (int j = 0; j < kernel_size; j++) {
            int input_j = in_y + j * dilation;
            if (input_j < 0 || input_j >= input_dim2) continue;
            
            for (int i = 0; i < kernel_size; i++) {
                int input_i = in_x + i * dilation;
                if (input_i < 0 || input_i >= input_dim1) continue;
                
                scalar_t val = input[batch * channels * input_dim1 * input_dim2 * input_dim3 +
                                    channel * input_dim1 * input_dim2 * input_dim3 +
                                    input_i * input_dim2 * input_dim3 +
                                    input_j * input_dim3 +
                                    input_k];
                
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    int output_offset = batch * channels * output_dim1 * output_dim2 * output_dim3 +
                       channel * output_dim1 * output_dim2 * output_dim3 +
                       idx * output_dim2 * output_dim3 +
                       idy * output_dim3 +
                       idz;
    output[output_offset] = max_val;
}

torch::Tensor max_pool3d_cuda(torch::Tensor input,
                             int kernel_size, int stride,
                             int padding, int dilation) {

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_dim1 = input.size(2);
    const auto input_dim2 = input.size(3);
    const auto input_dim3 = input.size(4);

    // Compute output dimensions
    auto output_dim1 = (input_dim1 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto output_dim2 = (input_dim2 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto output_dim3 = (input_dim3 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_dim1, output_dim2, output_dim3}, input.options());

    dim3 block(16, 16, 4);
    dim3 grid(
        (output_dim1 + block.x - 1) / block.x,
        (output_dim2 + block.y - 1) / block.y,
        (output_dim3 + block.z - 1) / block.z
    );

    grid.batch = batch_size;
    grid.warp = channels;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward", ([&] {
        max_pool3d_forward<scalar_t><<<grid, block>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, channels,
            input_dim1, input_dim2, input_dim3,
            output_dim1, output_dim2, output_dim3,
            kernel_size, stride, padding, dilation);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

maxpool3d_cpp_source = "torch::Tensor max_pool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);"

# Compile the CUDA kernel
max_pool3d_ext = load_inline(
    name='max_pool3d_cuda',
    cpp_sources=[maxpool3d_cpp_source],
    cuda_sources=[maxpool3d_source],
    functions=['max_pool3d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size, stride, padding, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.forward_func = max_pool3d_ext

    def forward(self, x):
        return self.forward_func.max_pool3d_cuda(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )