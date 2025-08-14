import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avg_pool_2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool2d_kernel(const scalar_t* input, scalar_t* output,
    int batch_size, int channels, int input_height, int input_width,
    int kernel_size, int stride, int pad_h, int pad_w,
    int output_height, int output_width) {

    const int batch_idx = blockIdx.x;
    const int channel_idx = blockIdx.y;
    const int out_y = threadIdx.y;
    const int out_x = threadIdx.x;

    const int thread_idx = out_y * blockDim.x + out_x;
    const int num_threads = blockDim.y * blockDim.x;

    for (int idx = thread_idx; idx < output_height * output_width; idx += num_threads) {
        const int out_y_global = idx / output_width;
        const int out_x_global = idx % output_width;

        const int in_y_start = out_y_global * stride - pad_h;
        const int in_x_start = out_x_global * stride - pad_w;
        const int in_y_end = in_y_start + kernel_size;
        const int in_x_end = in_x_start + kernel_size;

        scalar_t sum = 0.0;
        for (int in_y = in_y_start; in_y < in_y_end; ++in_y) {
            for (int in_x = in_x_start; in_x < in_x_end; ++in_x) {
                if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                    const int in_offset = (batch_idx * channels + channel_idx) * input_height * input_width
                        + in_y * input_width + in_x;
                    sum += input[in_offset];
                }
            }
        }

        const int out_offset = (batch_idx * channels + channel_idx) * output_height * output_width
            + out_y_global * output_width + out_x_global;
        output[out_offset] = sum / (kernel_size * kernel_size);
    }
}

torch::Tensor avg_pool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    dim3 block_dim(32, 8);  // Block size: 32x8 threads (adjust as needed)
    dim3 grid_dim(batch_size, channels);  // Each batch and channel processed in a separate block

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool2d_cuda", ([&] {
        avg_pool2d_kernel<scalar_t><<<grid_dim, block_dim>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, channels, input_height, input_width,
            kernel_size, stride, padding, padding,  // Assuming padding is symmetric
            output_height, output_width
        );
    }));

    return output;
}
"""

avg_pool_2d_cpp_source = """
torch::Tensor avg_pool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
"""

avg_pool_2d = load_inline(
    name="avg_pool_2d",
    cpp_sources=avg_pool_2d_cpp_source,
    cuda_sources=avg_pool_2d_source,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avg_pool_2d.avg_pool2d_cuda(x, self.kernel_size, self.stride, self.padding)

def get_inputs():
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size]