import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for MaxPool1d
maxpool1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename scalar_t>
__global__ void maxpool1d_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t batch_size,
    int64_t num_features,
    int64_t sequence_length,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices
) {
    int batch_idx = blockIdx.x;
    int feature_idx = blockIdx.y;
    int out_time_idx = threadIdx.x;

    int input_time_start = out_time_idx * stride - padding;
    int max_val = -INFINITY;
    int max_idx = -1;

    for (int kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
        int input_time = input_time_start + dilation * kernel_idx;
        if (input_time < 0 || input_time >= sequence_length) {
            continue;
        }
        int input_offset = batch_idx * num_features * sequence_length +
                           feature_idx * sequence_length +
                           input_time;
        scalar_t val = input[input_offset];
        if (val > max_val) {
            max_val = val;
            max_idx = input_time;
        }
    }

    int output_offset = batch_idx * num_features * (sequence_length - kernel_size + 1 + (2 * padding) / stride) +
                        feature_idx * ((sequence_length - kernel_size + 1 + (2 * padding) / stride)) +
                        out_time_idx;

    output[output_offset] = max_val;
}

at::Tensor maxpool1d_forward_cuda(const at::Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool return_indices) {
    const auto batch_size = input.size(0);
    const auto num_features = input.size(1);
    const auto sequence_length = input.size(2);

    auto output_length = (sequence_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = at::empty({batch_size, num_features, output_length}, input.options());

    dim3 blocks(batch_size, num_features);
    dim3 threads(output_length);

    int shared_mem = 0;

    maxpool1d_forward_kernel<float><<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_features,
        sequence_length,
        kernel_size,
        stride,
        padding,
        dilation,
        return_indices);

    return output;
}
"""

maxpool1d_cpp_source = """
at::Tensor maxpool1d_forward_cuda(const at::Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool return_indices);
"""

maxpool1d = load_inline(
    name="maxpool1d",
    cpp_sources=maxpool1d_cpp_source,
    cuda_sources=maxpool1d_source,
    functions=["maxpool1d_forward_cuda"],
    verbose=True,
    extra_cflags=["-D__STRICT_ANSI__"],
    extra_cuda_cflags=["--expt-relaxed-constexpr"]
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.maxpool = maxpool1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool.maxpool1d_forward_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices)