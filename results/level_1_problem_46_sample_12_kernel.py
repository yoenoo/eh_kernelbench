import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D Average Pooling
avg_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void avg_pool3d_kernel(const scalar_t* input, scalar_t* output,
    int batch_size, int channels, int in_depth, int in_height, int in_width,
    int kernel_size, int stride, int padding, int out_depth, int out_height, int out_width) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * channels * out_depth * out_height * out_width) {
        return;
    }

    const int w = idx % out_width;
    const int h = (idx / out_width) % out_height;
    const int d = (idx / (out_width * out_height)) % out_depth;
    const int c = (idx / (out_width * out_height * out_depth)) % channels;
    const int n = idx / (out_width * out_height * out_depth * channels);

    scalar_t sum = 0;
    int in_d_start = d * stride - padding;
    int in_h_start = h * stride - padding;
    int in_w_start = w * stride - padding;

    for (int kernel_d = 0; kernel_d < kernel_size; ++kernel_d) {
        int id = in_d_start + kernel_d;
        if (id < 0 || id >= in_depth) continue;

        for (int kernel_h = 0; kernel_h < kernel_size; ++kernel_h) {
            int ih = in_h_start + kernel_h;
            if (ih < 0 || ih >= in_height) continue;

            for (int kernel_w = 0; kernel_w < kernel_size; ++kernel_w) {
                int iw = in_w_start + kernel_w;
                if (iw < 0 || iw >= in_width) continue;
                
                sum += input[
                    n * channels * in_depth * in_height * in_width +
                    c * in_depth * in_height * in_width +
                    id * in_height * in_width +
                    ih * in_width +
                    iw
                ];
            }
        }
    }

    int kernel_count = 0;
    for (int kernel_d = 0; kernel_d < kernel_size; ++kernel_d) {
        int id = in_d_start + kernel_d;
        if (id < 0 || id >= in_depth) continue;

        for (int kernel_h = 0; kernel_h < kernel_size; ++kernel_h) {
            int ih = in_h_start + kernel_h;
            if (ih < 0 || ih >= in_height) continue;

            for (int kernel_w = 0; kernel_w < kernel_size; ++kernel_w) {
                int iw = in_w_start + kernel_w;
                if (iw < 0 || iw >= in_width) continue;
                kernel_count++;
            }
        }
    }

    output[idx] = sum / kernel_count;  // divide by actual kernel area that participated
}

std::vector<int64_t> output_shape(int64_t in_depth, int64_t in_height, int64_t in_width, int64_t kernel_size, int64_t stride, int64_t padding) {
    int64_t out_depth = (in_depth + 2 * padding - kernel_size) / stride + 1;
    int64_t out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int64_t out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    return {out_depth, out_height, out_width};
}

torch::Tensor avg_pool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    const auto in_depth = input.size(2);
    const auto in_height = input.size(3);
    const auto in_width = input.size(4);

    auto output_sizes = output_shape(in_depth, in_height, in_width, kernel_size, stride, padding);
    int out_depth = output_sizes[0];
    int out_height = output_sizes[1];
    int out_width = output_sizes[2];

    auto output = torch::empty({input.size(0), input.size(1), out_depth, out_height, out_width}, input.options());

    const int batch_size = input.size(0);
    const int channels = input.size(1);

    int total_elements = batch_size * channels * out_depth * out_height * out_width;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "avg_pool3d_cuda", ([&] {
        avg_pool3d_kernel<scalar_t><<<grid_size, block_size>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, channels, in_depth, in_height, in_width,
            kernel_size, stride, padding,
            out_depth, out_height, out_width
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

avg_pool3d_cpp_source = (
    "at::Tensor avg_pool3d_cuda(at::Tensor input, int kernel_size, int stride, int padding);"
)

# Compile the inline CUDA code for 3D Average Pooling
avg_pool3d = load_inline(
    name="avg_pool3d",
    cuda_sources=avg_pool3d_source,
    cpp_sources=avg_pool3d_cpp_source,
    functions=["avg_pool3d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avg_pool3d.avg_pool3d_cuda(x.cuda(), self.kernel_size, self.stride, self.padding)

# Note: The provided get_inputs and get_init_inputs are reused from original model, but modified as needed