import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Average Pooling
avg_pool_2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

template <typename scalar_t>
__global__ void avg_pool_2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int out_height,
    const int out_width
) {
    const int batch = blockIdx.x;
    const int channel = blockIdx.y;
    const int out_y = threadIdx.x;
    const int out_x = threadIdx.y;

    scalar_t sum = 0;
    int count = 0;

    const int in_y_start = -pad_h + (out_y * stride_h);
    const int in_x_start = -pad_w + (out_x * stride_w);

    for (int ky = 0; ky < kernel_h; ++ky) {
        const int in_y = in_y_start + ky;
        if (in_y < 0 || in_y >= in_height) continue;
        for (int kx = 0; kx < kernel_w; ++kx) {
            const int in_x = in_x_start + kx;
            if (in_x < 0 || in_x >= in_width) continue;
            sum += input[batch * channels * in_height * in_width +
                        channel * in_height * in_width +
                        in_y * in_width + in_x];
            count++;
        }
    }

    if (count > 0) {
        output[batch * channels * out_height * out_width +
               channel * out_height * out_width +
               out_y * out_width + out_x] = sum / static_cast<scalar_t>(count);
    }
}

std::vector<int> compute_output_size(
    int input_height,
    int input_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    int output_height = (input_height + 2 * pad_h - kernel_h) / stride_h + 1;
    int output_width = (input_width + 2 * pad_w - kernel_w) / stride_w + 1;
    return {output_height, output_width};
}

template <typename scalar_t>
torch::Tensor avg_pool_2d_cuda(torch::Tensor input,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);

    auto output_sizes = compute_output_size(
        in_height,
        in_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w
    );
    const int out_height = output_sizes[0];
    const int out_width = output_sizes[1];

    auto output = torch::empty({batch_size, channels, out_height, out_width}, 
                                dtype(input.scalar_type()), 
                                input.device());

    dim3 blocks(batch_size, channels);
    dim3 threads(out_height, out_width);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool_2d_cuda", ([&]{
        avg_pool_2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            in_height,
            in_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_height,
            out_width
        );
    }));

    cudaDeviceSynchronize();
    return output;
}

torch::Tensor avg_pool_2d(torch::Tensor input,
                         int kernel_size_h,
                         int kernel_size_w,
                         int stride_h,
                         int stride_w,
                         int padding_h,
                         int padding_w
) {
    return avg_pool_2d_cuda(input, kernel_size_h, kernel_size_w,
                           stride_h, stride_w, padding_h, padding_w);
}
"""

avg_pool_2d_cpp_source = """
torch::Tensor avg_pool_2d(torch::Tensor input,
                        int kernel_size_h,
                        int kernel_size_w,
                        int stride_h,
                        int stride_w,
                        int padding_h,
                        int padding_w);
"""

# Compile the inline CUDA code
avg_pool_2d_op = load_inline(
    name="avg_pool_2d",
    cpp_sources=avg_pool_2d_cpp_source,
    cuda_sources=avg_pool_2d_source,
    functions=["avg_pool_2d"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.avg_pool_op = avg_pool_2d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool_op(
            x,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
        )