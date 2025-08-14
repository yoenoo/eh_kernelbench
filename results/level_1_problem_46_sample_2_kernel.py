import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avg_pool_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool_3d_kernel(
    const float* input, float* output,
    int batch_size, int channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int kernel_size, int stride, int padding) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * out_depth * out_height * out_width) return;

    // Compute indices for batch, channel, depth, height, width
    int output_w = idx % out_width;
    int output_h = (idx / out_width) % out_height;
    int output_d = (idx / (out_width * out_height)) % out_depth;
    int channel = (idx / (out_width * out_height * out_depth)) % channels;
    int batch = idx / (channels * out_depth * out_height * out_width);

    int in_start_d = output_d * stride - padding;
    int in_end_d = in_start_d + kernel_size;
    int in_start_h = output_h * stride - padding;
    int in_end_h = in_start_h + kernel_size;
    int in_start_w = output_w * stride - padding;
    int in_end_w = in_start_w + kernel_size;

    float sum = 0.0;
    int valid_count = 0;

    for (int d = in_start_d; d < in_end_d; ++d) {
        for (int h = in_start_h; h < in_end_h; ++h) {
            for (int w = in_start_w; w < in_end_w; ++w) {
                if (d >= 0 && d < in_depth &&
                    h >= 0 && h < in_height &&
                    w >= 0 && w < in_width) {
                    int in_idx = batch * channels * in_depth * in_height * in_width +
                                channel * in_depth * in_height * in_width +
                                d * in_height * in_width +
                                h * in_width +
                                w;
                    sum += input[in_idx];
                    valid_count++;
                }
            }
        }
    }

    output[idx] = sum / valid_count;
}

torch::Tensor avg_pool_3d_cuda(
    torch::Tensor input,
    int kernel_size, int stride, int padding) {

    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    // Calculate output dimensions
    int out_depth = (in_depth + 2 * padding - kernel_size) / stride + 1;
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, channels, out_depth, out_height, out_width}, input.options());

    int total_elements = batch_size * channels * out_depth * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    avg_pool_3d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_size, stride, padding);

    return output;
}
"""

avg_pool_3d_cpp_source = (
    "torch::Tensor avg_pool_3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"
)

avg_pool_3d = load_inline(
    name="avg_pool_3d",
    cpp_sources=avg_pool_3d_cpp_source,
    cuda_sources=avg_pool_3d_source,
    functions=["avg_pool_3d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avg_pool_3d.avg_pool_3d_cuda(
            x, self.kernel_size, self.stride, self.padding
        )