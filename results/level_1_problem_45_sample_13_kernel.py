import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D Average Pooling
avg_pool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool2d_kernel(const scalar_t* input, scalar_t* output,
                                 int batch_size, int channels, int input_height, int input_width,
                                 int kernel_size, int output_height, int output_width, int padding, int stride) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * channels * output_height * output_width) {
        return;
    }

    const int w = idx % output_width;
    const int h = (idx / output_width) % output_height;
    const int c = (idx / output_width / output_height) % channels;
    const int n = idx / output_width / output_height / channels;

    scalar_t sum = 0;
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            int h_in = h * stride + ky - padding;
            int w_in = w * stride + kx - padding;

            if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                sum += input[((n * channels + c) * input_height + h_in) * input_width + w_in];
            }
        }
    }
    output[idx] = sum / (kernel_size * kernel_size);
}

std::vector<int64_t> output_size(int64_t batch_size, int64_t channels, int64_t input_height, int64_t input_width,
                                int64_t kernel_size, int64_t stride, int64_t padding) {
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    return {batch_size, channels, output_height, output_width};
}

torch::Tensor avg_pool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    if (stride == 0) {
        stride = kernel_size;
    }

    auto output_size_vec = output_size(batch_size, channels, input_height, input_width, kernel_size, stride, padding);
    auto output_height = output_size_vec[2];
    auto output_width = output_size_vec[3];

    auto output = torch::empty(output_size_vec, input.options());

    const int threads_per_block = 256;
    const int num_elements = batch_size * channels * output_height * output_width;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "avg_pool2d_cuda", ([&] {
        avg_pool2d_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, channels, input_height, input_width,
            kernel_size, output_height, output_width, padding, stride
        );
    }));

    return output;
}
"""

avg_pool2d_cpp_source = """
#include <torch/extension.h>

std::vector<int64_t> output_size(int64_t batch_size, int64_t channels, int64_t input_height, int64_t input_width,
                                int64_t kernel_size, int64_t stride, int64_t padding);
torch::Tensor avg_pool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code for 2D Average Pooling
avg_pool2d = load_inline(
    name="avg_pool2d",
    cpp_sources=avg_pool2d_cpp_source,
    cuda_sources=avg_pool2d_source,
    functions=["avg_pool2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.avg_pool_cuda = avg_pool2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool_cuda.avg_pool2d_cuda(x, self.kernel_size, self.stride, self.padding)