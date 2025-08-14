import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avg_pool1d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool1d_forward_kernel(const scalar_t* input, scalar_t* output,
                          int batch_size, int channels, int in_length, int out_length,
                          int kernel_size, int padding, int stride) {

    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int out_idx = threadIdx.x;

    int in_start = out_idx * stride - padding;
    int in_end = in_start + kernel_size;

    scalar_t sum = 0.0;
    int valid_count = 0;

    for (int i = in_start; i < in_end; ++i) {
        if (i >=0 && i < in_length) {
            sum += input[batch_idx * channels * in_length + channel_idx * in_length + i];
            valid_count++;
        }
    }

    if (valid_count > 0) {
        output[batch_idx * channels * out_length + channel_idx * out_length + out_idx] = sum / valid_count;
    }
}

torch::Tensor avg_pool1d_forward_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    auto output_size = input.size(0), input_channels = input.size(1);
    int in_length = input.size(2);
    int out_length = (in_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({output_size, input_channels, out_length}, 
                              input.options());

    dim3 threads(out_length);
    dim3 blocks(input.size(0), input_channels);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "avg_pool1d_forward", ([&] {
        avg_pool1d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.size(0), input.size(1), in_length, out_length,
            kernel_size, padding, stride
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

avg_pool1d_cpp_source = "torch::Tensor avg_pool1d_forward_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"

avg_pool1d = load_inline(
    name="avg_pool1d",
    cpp_sources=avg_pool1d_cpp_source,
    cuda_sources=avg_pool1d_source,
    functions=["avg_pool1d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool1d_forward_cuda = avg_pool1d

    def forward(self, x):
        return self.avg_pool1d_forward_cuda.avg_pool1d_forward_cuda(
            x, self.kernel_size, self.stride, self.padding
        )