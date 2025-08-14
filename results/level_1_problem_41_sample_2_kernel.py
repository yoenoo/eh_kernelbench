import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_pool1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void max_pool1d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int num_features,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int index = idx; index < batch_size * num_features * out_length; index += total_threads) {
        int seq_idx = index % out_length;
        int feature_idx = (index / out_length) % num_features;
        int batch_idx = index / (num_features * out_length);

        int in_start = seq_idx * stride - padding;
        int max_val = -INFINITY;
        for (int k = 0; k < kernel_size; ++k) {
            int pos = in_start + k * dilation;
            if (pos < 0 || pos >= in_length) {
                continue;
            }
            scalar_t val = input[batch_idx * num_features * in_length + feature_idx * in_length + pos];
            if (val > max_val) {
                max_val = val;
            }
        }
        output[index] = max_val;
    }
}

torch::Tensor max_pool1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int in_length = input.size(2);

    int out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, num_features, out_length}, input.options());

    int threads = 512;
    int blocks = (batch_size * num_features * out_length + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool1d_cuda", ([&] {
        max_pool1d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_features,
            in_length,
            out_length,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

max_pool1d_cpp_source = "torch::Tensor max_pool1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);"

max_pool1d = load_inline(
    name="max_pool1d",
    cpp_sources=max_pool1d_cpp_source,
    cuda_sources=max_pool1d_source,
    functions=["max_pool1d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.max_pool1d = max_pool1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool1d.max_pool1d_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)