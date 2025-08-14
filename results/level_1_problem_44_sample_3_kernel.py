import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avg_pool1d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(const float* input, float* output,
                                 int batch_size, int channels, int input_length,
                                 int kernel_size, int output_length,
                                 int padding, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elements = batch_size * channels * output_length;

    if (idx >= num_elements) return;

    int channel = (idx / output_length) % channels;
    int batch = idx / (channels * output_length);
    int out_pos = idx % output_length;

    int in_start = out_pos * stride - padding;
    int in_end = in_start + kernel_size;

    float sum = 0.0;
    for (int i = in_start; i < in_end; ++i) {
        if (i >= 0 && i < input_length) {
            sum += input[batch * channels * input_length + channel * input_length + i];
        }
    }
    output[idx] = sum / kernel_size;
}

torch::Tensor avg_pool1d_cuda(torch::Tensor input,
                             int kernel_size, int stride, int padding) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_length = input.size(2);
    
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    auto output = torch::empty({batch_size, channels, output_length}, input.options());

    int block_size = 256;
    int num_threads = batch_size * channels * output_length;
    int num_blocks = (num_threads + block_size - 1) / block_size;

    avg_pool1d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, input_length,
        kernel_size, output_length,
        padding, stride
    );

    return output;
}

"""

avg_pool1d_cpp_source = "torch::Tensor avg_pool1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"

avg_pool_cuda = load_inline(
    name="avg_pool_cuda",
    cpp_sources=avg_pool1d_cpp_source,
    cuda_sources=avg_pool1d_source,
    functions=["avg_pool1d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.forward_cuda = avg_pool_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_cuda.avg_pool1d_cuda(x, self.kernel_size, self.stride, self.padding)