import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 2D Average Pooling
avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool2d_kernel(const scalar_t* input, scalar_t* output,
                                 int batch, int channels, int input_height, int input_width,
                                 int kernel_size, int stride, int padding,
                                 int output_height, int output_width) {
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int out_y = threadIdx.x;
    int out_x = threadIdx.y;

    int output_size = output_height * output_width;
    int idx = out_y * output_width + out_x;
    int input_y = -padding + out_y * stride;
    int input_x = -padding + out_x * stride;

    scalar_t sum = 0.0;
    int cnt = 0;

    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            int y = input_y + ky;
            int x = input_x + kx;
            if (y >= 0 && y < input_height && x >= 0 && x < input_width) {
                int input_idx = batch_idx * channels * input_height * input_width +
                                channel_idx * input_height * input_width +
                                y * input_width + x;
                sum += input[input_idx];
                ++cnt;
            }
        }
    }
    
    int output_idx = batch_idx * channels * output_size +
                     channel_idx * output_size +
                     out_y * output_width + out_x;
    output[output_idx] = sum / static_cast<scalar_t>(cnt);
}

std::tuple<torch::Tensor> avg_pool2d_cuda(torch::Tensor input,
                                          int kernel_size, int stride, int padding) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    const int output_height = (input_height + 2 * padding -
                              kernel_size) / stride + 1;
    const int output_width = (input_width + 2 * padding -
                             kernel_size) / stride + 1;

    auto output = torch::empty({batch, channels, output_height, output_width},
                              dtype(input.dtype()), device(input.device()));

    dim3 threads(kernel_size, kernel_size);
    dim3 blocks(batch, channels);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "avg_pool2d_cuda", ([&] {
        avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch, channels, input_height, input_width,
            kernel_size, stride, padding,
            output_height, output_width);
    }));

    return output;
}
"""

avg_pool_cpp_source = """
std::tuple<torch::Tensor> avg_pool2d_cuda(torch::Tensor input,
                                          int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code
avg_pool = load_inline(
    name='avg_pool_cuda',
    cpp_sources=avg_pool_cpp_source,
    cuda_sources=avg_pool_source,
    functions=['avg_pool2d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.forward_op = avg_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_op.avg_pool2d_cuda(x, self.kernel_size, self.stride, self.padding)[0]