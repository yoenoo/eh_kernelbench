import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

        # Define and load the custom CUDA kernel
        avg_pool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void avg_pool2d_kernel(const T* input, T* output, int batch_size, int channels,
                                 int input_height, int input_width, int kernel_size,
                                 int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate output dimensions
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    for (; idx < batch_size * channels * output_height * output_width; idx += blockDim.x * gridDim.x) {
        // Calculate output position
        int w = idx % output_width;
        int h = (idx / output_width) % output_height;
        int c = (idx / (output_width * output_height)) % channels;
        int n = idx / (output_width * output_height * channels);

        // Calculate input region
        int in_h_start = h * stride - padding;
        int in_w_start = w * stride - padding;
        int in_h_end = in_h_start + kernel_size;
        int in_w_end = in_w_start + kernel_size;

        T sum = static_cast<T>(0);
        int count = 0;

        for (int iy = in_h_start; iy < in_h_end; ++iy) {
            for (int ix = in_w_start; ix < in_w_end; ++ix) {
                if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width) {
                    int in_idx = n * channels * input_height * input_width +
                                c * input_height * input_width +
                                iy * input_width + ix;
                    sum += input[in_idx];
                    count++;
                }
            }
        }
        output[idx] = sum / static_cast<T>(count);
    }
}

std::tuple<torch::Tensor> avg_pool2d_cuda(torch::Tensor input, int kernel_size,
                                        int stride, int padding) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    auto options = torch::TensorOptions().like(input);
    auto output = torch::empty({batch_size, channels, output_height, output_width}, options);

    int threads = 512;
    int elements = batch_size * channels * output_height * output_width;
    int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool2d_cuda", ([&] {
        avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, channels,
            input_height, input_width,
            kernel_size, stride, padding
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

        avg_pool2d_cpp_source = (
            "at::Tensor avg_pool2d_cuda(at::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding);"
        )

        self.avg_pool_cuda = load_inline(
            name="avg_pool_cuda",
            cpp_sources=avg_pool2d_cpp_source,
            cuda_sources=avg_pool2d_source,
            functions=["avg_pool2d_cuda"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to CUDA tensors and run custom kernel
        x = x.cuda()
        return self.avg_pool_cuda.avg_pool2d_cuda(x, self.kernel_size, self.stride, self.padding)