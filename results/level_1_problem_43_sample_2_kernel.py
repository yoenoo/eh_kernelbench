cuda
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for MaxPool3d
maxpool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename scalar_t>
__global__ void maxpool3d_forward_kernel(const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> input,
                                        torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> output,
                                        torch::PackedTensorAccessor<int, 5, torch::RestrictPtrTraits> indices,
                                        const int kernel_size,
                                        const int stride,
                                        const int padding,
                                        const int dilation,
                                        const int batch_size,
                                        const int channels,
                                        const int in_depth,
                                        const int in_height,
                                        const int in_width,
                                        const int out_depth,
                                        const int out_height,
                                        const int out_width) {
    const int d = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel = blockIdx.y % channels;
    const int batch = blockIdx.x / (out_depth * out_height * out_width);

    if (batch >= batch_size || channel >= channels || d >= out_depth || h >= out_height || w >= out_width) {
        return;
    }

    const int in_d_start = d * stride - padding;
    const int in_h_start = h * stride - padding;
    const int in_w_start = w * stride - padding;

    scalar_t max_val = -FLT_MAX;
    int max_idx = 0;
    int idx_count = 0;

    for (int kd = 0; kd < kernel_size; ++kd) {
        const int id = in_d_start + kd * dilation;
        if (id < 0 || id >= in_depth) continue;

        for (int kh = 0; kh < kernel_size; ++kh) {
            const int ih = in_h_start + kh * dilation;
            if (ih < 0 || ih >= in_height) continue;

            for (int kw = 0; kw < kernel_size; ++kw) {
                const int iw = in_w_start + kw * dilation;
                if (iw < 0 || iw >= in_width) continue;

                const scalar_t val = input[batch][channel][id][ih][iw];
                if (val > max_val) {
                    max_val = val;
                    max_idx = idx_count;
                }
                idx_count++;
            }
        }
    }

    output[batch][channel][d][h][w] = max_val;
    indices[batch][channel][d][h][w] = max_idx;
}

at::Tensor maxpool3d_forward_cuda(const at::Tensor& input, int kernel_size, int stride, int padding, int dilation) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    // Compute output dimensions
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    at::Tensor output = at::empty({batch_size, channels, out_depth, out_height, out_width}, input.options());
    at::Tensor indices = at::empty({batch_size, channels, out_depth, out_height, out_width}, input.options().dtype(at::kInt));

    dim3 block(16, 16, 1);
    dim3 grid(1, 1, 1);

    // Adjust the block and grid dimensions
    grid.x = (out_width + block.x - 1) / block.x;
    grid.y = (out_height + block.y - 1) / block.y;
    grid.z = out_depth * channels * batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "maxpool3d_forward", ([&] {
        maxpool3d_forward_kernel<scalar_t><<<grid, block>>>(
            input.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits>(),
            indices.packed_accessor<int, 5, torch::RestrictPtrTraits>(),
            kernel_size,
            stride,
            padding,
            dilation,
            batch_size,
            channels,
            in_depth,
            in_height,
            in_width,
            out_depth,
            out_height,
            out_width
        );
    }));

    return output;
}
"""

# Compile the inline CUDA code
maxpool3d_forward = load_inline(
    name="maxpool3d_forward",
    cpp_sources="",
    cuda_sources=maxpool3d_source,
    functions=["maxpool3d_forward_cuda"],
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
        self.maxpool3d_forward_cuda = maxpool3d_forward.maxpool3d_forward_cuda

    def forward(self, x):
        output = self.maxpool3d_forward_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)
        return output