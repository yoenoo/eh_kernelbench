import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D max pooling
maxpool3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void maxpool3d_forward_kernel(const scalar_t* __restrict__ input,
                                        scalar_t* __restrict__ output,
                                        const int batch_size,
                                        const int channels,
                                        const int input_dim1,
                                        const int input_dim2,
                                        const int input_dim3,
                                        const int output_dim1,
                                        const int output_dim2,
                                        const int output_dim3,
                                        const int kernel_size,
                                        const int stride,
                                        const int padding,
                                        const int dilation,
                                        const int ceil_mode) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel_stride = input_dim1 * input_dim2 * input_dim3;
    const int output_size = batch_size * channels * output_dim1 * output_dim2 * output_dim3;

    if (idx >= output_size) return;

    const int batch = idx / (channels * output_dim1 * output_dim2 * output_dim3);
    const int c = (idx / (output_dim1 * output_dim2 * output_dim3)) % channels;
    const int out_d1 = (idx / (output_dim2 * output_dim3)) % output_dim1;
    const int out_d2 = (idx / output_dim3) % output_dim2;
    const int out_d3 = idx % output_dim3;

    int in_d1 = -padding + (out_d1) * stride;
    int in_d2 = -padding + (out_d2) * stride;
    int in_d3 = -padding + (out_d3) * stride;

    scalar_t max_val = -FLT_MAX;

    for (int k1 = 0; k1 < kernel_size; ++k1) {
        for (int k2 = 0; k2 < kernel_size; ++k2) {
            for (int k3 = 0; k3 < kernel_size; ++k3) {
                int d1 = in_d1 + k1 * dilation;
                int d2 = in_d2 + k2 * dilation;
                int d3 = in_d3 + k3 * dilation;

                // Check if the current position is within input boundaries
                if (d1 >= 0 && d1 < input_dim1 &&
                    d2 >= 0 && d2 < input_dim2 &&
                    d3 >= 0 && d3 < input_dim3) {
                    int input_idx = batch * channel_stride +
                        c * input_dim1 * input_dim2 * input_dim3 +
                        d1 * input_dim2 * input_dim3 +
                        d2 * input_dim3 +
                        d3;
                    scalar_t val = input[input_idx];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }
    }

    output[idx] = max_val;
}

std::vector<int64_t> calculate_output_dims(int input_size, int kernel_size,
                                          int stride, int padding, int dilation,
                                          bool ceil_mode) {
    int numerator = input_size + 2 * padding - dilation * (kernel_size - 1) - 1;
    int kernel_effective = (kernel_size - 1) * dilation + 1;
    if (ceil_mode) {
        return {(numerator + stride - 1) / stride};
    } else {
        return {numerator / stride + 1};
    }
}

torch::Tensor maxpool3d_forward(torch::Tensor input,
                                int kernel_size,
                                int stride,
                                int padding,
                                int dilation,
                                bool return_indices,
                                bool ceil_mode) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_dim1 = input.size(2);
    const auto input_dim2 = input.size(3);
    const auto input_dim3 = input.size(4);

    // Calculate output dimensions
    auto output_dim1 = calculate_output_dims(input_dim1, kernel_size, stride, padding, dilation, ceil_mode)[0];
    auto output_dim2 = calculate_output_dims(input_dim2, kernel_size, stride, padding, dilation, ceil_mode)[0];
    auto output_dim3 = calculate_output_dims(input_dim3, kernel_size, stride, padding, dilation, ceil_mode)[0];

    auto output_size = {batch_size, channels, output_dim1, output_dim2, output_dim3};
    auto output = torch::empty(output_size, input.options());

    const int threads = 256;
    const int blocks = (batch_size * channels * output_dim1 * output_dim2 * output_dim3 + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool3d_forward", ([&] {
        maxpool3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_dim1,
            input_dim2,
            input_dim3,
            output_dim1,
            output_dim2,
            output_dim3,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode ? 1 : 0
        );
    }));

    return output;
}
"""

maxpool3d_cpp_source = """
torch::Tensor maxpool3d_forward(torch::Tensor input,
                                int kernel_size,
                                int stride,
                                int padding,
                                int dilation,
                                bool return_indices,
                                bool ceil_mode);
"""

# Compile the CUDA extension
maxpool3d = load_inline(
    name='maxpool3d',
    cpp_sources=maxpool3d_cpp_source,
    cuda_sources=maxpool3d_source,
    functions=['maxpool3d_forward'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.maxpool3d = maxpool3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool3d.maxpool3d_forward(
            x.cuda(), 
            self.kernel_size, 
            self.stride, 
            self.padding,
            self.dilation,
            self.return_indices,
            self.ceil_mode
        )