import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D Average Pooling
avg_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

template <typename scalar_t>
__global__ void avg_pool3d_kernel(const torch::PackedTensorAccessor<scalar_t,5> input,
                                 torch::PackedTensorAccessor<scalar_t,5> output,
                                 int kernel_size,
                                 int stride,
                                 int padding) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    const int out_depth = output.size(2);
    const int out_height = output.size(3);
    const int out_width = output.size(4);

    int batch = blockIdx.x;
    int channel = blockIdx.y;
    int d = blockIdx.z;
    int h = threadIdx.y;
    int w = threadIdx.x;

    if (batch >= batch_size || channel >= channels)
        return;

    int in_d_start = d * stride - padding;
    int in_h_start = h * stride - padding;
    int in_w_start = w * stride - padding;

    scalar_t sum = 0;
    int count = 0;

    for (int kd = 0; kd < kernel_size; ++kd) {
        int current_d = in_d_start + kd;
        if (current_d < 0 || current_d >= in_depth)
            continue;
        for (int kh = 0; kh < kernel_size; ++kh) {
            int current_h = in_h_start + kh;
            if (current_h < 0 || current_h >= in_height)
                continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int current_w = in_w_start + kw;
                if (current_w < 0 || current_w >= in_width)
                    continue;
                sum += input[batch][channel][current_d][current_h][current_w];
                count++;
            }
        }
    }
    if (count > 0) {
        output[batch][channel][d][h][w] = sum / count;
    }
}

std::vector<int64_t> output_size(std::vector<int64_t> input_size,
                                int kernel_size,
                                int stride,
                                int padding) {
    std::vector<int64_t> output_dims(5);
    output_dims[0] = input_size[0];
    output_dims[1] = input_size[1];
    output_dims[2] = (input_size[2] + 2 * padding - kernel_size) / stride + 1;
    output_dims[3] = (input_size[3] + 2 * padding - kernel_size) / stride + 1;
    output_dims[4] = (input_size[4] + 2 * padding - kernel_size) / stride + 1;
    return output_dims;
}

torch::Tensor avg_pool3d_cuda(torch::Tensor input,
                             int kernel_size,
                             int stride,
                             int padding) {
    auto input_size = input.sizes().vec();
    auto output_dims = output_size(input_size, kernel_size, stride, padding);
    auto output = torch::empty(output_dims, input.options());

    int block_dim_x = 32;
    int block_dim_y = 8;
    dim3 block(block_dim_x, block_dim_y);
    dim3 grid(input_size[0], input_size[1], output_dims[2]);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool3d_cuda", ([&] {
        avg_pool3d_kernel<scalar_t><<<grid, block>>>(
            input.packed_accessor<scalar_t,5>(),
            output.packed_accessor<scalar_t,5>(),
            kernel_size,
            stride,
            padding);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

avg_pool3d_cpp_source = """
torch::Tensor avg_pool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code for custom 3D Average Pooling
avg_pool3d = load_inline(
    name="avg_pool3d",
    cpp_sources=avg_pool3d_cpp_source,
    cuda_sources=avg_pool3d_source,
    functions=["avg_pool3d_cuda"],
    verbose=True,
    extra_cflags=["-DENABLE_cuda"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.avg_pool = avg_pool3d  # replaced with custom CUDA operator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool.avg_pool3d_cuda(x, self.kernel_size, self.stride, self.padding)