import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avg_pool1d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void avg_pool1d_forward_kernel(const scalar_t* __restrict__ input,
                                         scalar_t* __restrict__ output,
                                         const int batch_size,
                                         const int in_channels,
                                         const int input_length,
                                         const int kernel_size,
                                         const int stride,
                                         const int padding) {
    const int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_channels * output_length) return;

    const int channel = (idx / output_length) % in_channels;
    const int batch = idx / (in_channels * output_length);
    const int out_pos = idx % output_length;

    int in_start = out_pos * stride - padding;
    in_start = max(in_start, 0);
    int in_end = in_start + kernel_size;
    in_end = min(in_end, input_length);

    scalar_t sum = 0;
    for (int i = in_start; i < in_end; ++i) {
        sum += input[batch * in_channels * input_length + channel * input_length + i];
    }
    output[idx] = sum / static_cast<scalar_t>(kernel_size);
}

torch::Tensor avg_pool1d_forward(torch::Tensor input, int kernel_size, int stride, int padding) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    auto output = torch::empty({batch_size, in_channels, output_length}, input.options());

    const int threads = 256;
    const int elements = batch_size * in_channels * output_length;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "avg_pool1d_forward", ([&] {
        avg_pool1d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            input_length,
            kernel_size,
            stride,
            padding
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

avg_pool1d_header = """
torch::Tensor avg_pool1d_forward(torch::Tensor input, int kernel_size, int stride, int padding);
"""

avg_pool = load_inline(
    name="avg_pool1d",
    cpp_sources=avg_pool1d_header,
    cuda_sources=avg_pool1d_source,
    functions=["avg_pool1d_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avg_pool.avg_pool1d_forward(x.cuda(), self.kernel_size, self.stride, self.padding).cuda()