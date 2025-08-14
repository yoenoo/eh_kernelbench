import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D Average Pooling
avg_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool3d_kernel(
    const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits> output,
    int kernel_size,
    int stride,
    int padding,
    int batch_size,
    int channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * channels * out_depth * out_height * out_width) {
        return;
    }

    const int w = idx % out_width;
    const int h = (idx / out_width) % out_height;
    const int d = (idx / (out_width * out_height)) % out_depth;
    const int c = (idx / (out_width * out_height * out_depth)) % channels;
    const int n = idx / (out_width * out_height * out_depth * channels);

    const int in_offset_d = d * stride - padding;
    const int in_offset_h = h * stride - padding;
    const int in_offset_w = w * stride - padding;

    scalar_t sum = 0;
    int count = 0;

    for (int kd = 0; kd < kernel_size; ++kd) {
        const int id = in_offset_d + kd;
        if (id < 0 || id >= in_depth) continue;
        for (int kh = 0; kh < kernel_size; ++kh) {
            const int ih = in_offset_h + kh;
            if (ih < 0 || ih >= in_height) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int iw = in_offset_w + kw;
                if (iw < 0 || iw >= in_width) continue;
                sum += input[n][c][id][ih][iw];
                count++;
            }
        }
    }

    if (count > 0) {
        output[n][c][d][h][w] = sum / static_cast<scalar_t>(count);
    }
}

std::tuple<torch::Tensor> avg_pool3d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    int out_depth = (in_depth + 2 * padding - kernel_size) / stride + 1;
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto options = torch::TensorOptions().like(input);
    torch::Tensor output = torch::empty({batch_size, channels, out_depth, out_height, out_width}, options);

    const int num_elements = batch_size * channels * out_depth * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    const int threads_per_block = block_size;
    const dim3 blocks(num_blocks);
    const dim3 threads(threads_per_block);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool3d_cuda", ([&] {
        avg_pool3d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits>(),
            kernel_size,
            stride,
            padding,
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

    cudaDeviceSynchronize();
    return std::make_tuple(output);
}
"""

avg_pool3d_cpp_source = (
    "std::tuple<torch::Tensor> avg_pool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"
)

# Compile the inline CUDA code
avg_pool3d = load_inline(
    name="avg_pool3d",
    cpp_sources=avg_pool3d_cpp_source,
    cuda_sources=avg_pool3d_source,
    functions=["avg_pool3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.avg_pool_cuda = avg_pool3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool_cuda.avg_pool3d_cuda(
            x, self.kernel_size, self.stride, self.padding)[0]