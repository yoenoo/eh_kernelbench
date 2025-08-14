import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for MaxPool2d
maxpool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

struct MaxPool2DConfig {
    int kernel_size;
    int stride;
    int padding;
    int dilation;
};

__global__ void max_pool_2d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    MaxPool2DConfig config) {

    const int batch = blockIdx.x;
    const int channel = blockIdx.y;
    const int out_y = threadIdx.y;
    const int out_x = threadIdx.x + blockDim.x * blockIdx.z;

    if (out_x >= output_width) return;

    const int kernel_size = config.kernel_size;
    const int stride = config.stride;
    const int padding = config.padding;
    const int dilation = config.dilation;

    int max_val = -FLT_MAX;
    int in_x_start = out_x * stride - padding;
    int in_y_start = out_y * stride - padding;

    for (int dy = 0; dy < kernel_size; ++dy) {
        int in_y = in_y_start + dy * dilation;
        if (in_y < 0 || in_y >= input_height) continue;

        for (int dx = 0; dx < kernel_size; ++dx) {
            int in_x = in_x_start + dx * dilation;
            if (in_x < 0 || in_x >= input_width) continue;

            int input_idx = batch * channels * input_height * input_width +
                            channel * input_height * input_width +
                            in_y * input_width + in_x;

            float current_val = input[input_idx];
            if (current_val > max_val) {
                max_val = current_val;
            }
        }
    }

    int output_idx = batch * channels * output_height * output_width +
                     channel * output_height * output_width +
                     out_y * output_width + out_x;

    output[output_idx] = max_val;
}

torch::Tensor max_pool_2d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    MaxPool2DConfig config = {
        .kernel_size = kernel_size,
        .stride = stride,
        .padding = padding,
        .dilation = dilation
    };

    dim3 block_dim(256, 1, 1); // Threads per block
    dim3 grid_dim(batch_size, channels, output_width); // Blocks per grid

    max_pool_2d_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, input_height, input_width,
        output_height, output_width,
        config);

    return output;
}
"""

maxpool2d_cpp_source = """
torch::Tensor max_pool_2d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation);
"""

# Compile the inline CUDA code
max_pool_2d = load_inline(
    name="max_pool_2d",
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_source,
    functions=["max_pool_2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.max_pool_2d = max_pool_2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool_2d.max_pool_2d_cuda(
            x, self.kernel_size, self.stride, self.padding, self.dilation)