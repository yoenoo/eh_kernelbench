import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_pool1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void max_pool1d_cuda_forward(const float* input, float* output, int batch_size, int features, int in_length, int out_length, int kernel_size, int stride, int padding, int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;

    int batch = idx / features;
    int feature = idx % features;

    int in_offset = batch * features * in_length + feature * in_length;
    int out_offset = batch * features * out_length + feature * out_length;

    for (int out_pos = 0; out_pos < out_length; ++out_pos) {
        float max_val = -FLT_MAX;
        int max_idx = -1;
        int start = out_pos * stride - padding;
        for (int k = 0; k < kernel_size; ++k) {
            int in_pos = start + k * dilation;
            if (in_pos < 0 || in_pos >= in_length) continue;
            float val = input[in_offset + in_pos];
            if (val > max_val) {
                max_val = val;
            }
        }
        output[out_offset + out_pos] = max_val;
    }
}

torch::Tensor max_pool1d_forward_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    const int threads = 256;
    const int batches_features = input.size(0) * input.size(1);
    const int blocks = (batches_features + threads - 1) / threads;

    int in_length = input.size(2);
    int out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({input.size(0), input.size(1), out_length}, input.options());

    max_pool1d_cuda_forward<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0),
        input.size(1),
        in_length,
        out_length,
        kernel_size,
        stride,
        padding,
        dilation
    );

    cudaDeviceSynchronize();
    return output;
}
"""

max_pool1d_cpp_source = "torch::Tensor max_pool1d_forward_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);"

max_pool1d = load_inline(
    name="max_pool1d",
    cpp_sources=[max_pool1d_cpp_source],
    cuda_sources=[max_pool1d_source],
    functions=["max_pool1d_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.max_pool1d = max_pool1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool1d.max_pool1d_forward_cuda(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )