import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

maxpool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool3d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_dim1,
    const int in_dim2,
    const int in_dim3,
    const int out_dim1,
    const int out_dim2,
    const int out_dim3,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const bool ceil_mode) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * out_dim1 * out_dim2 * out_dim3) return;

    const int d_o = idx % out_dim3;
    const int c = idx / (out_dim1 * out_dim2 * out_dim3);
    const int rem = idx / (out_dim2 * out_dim3) % out_dim1;
    const int h_o = rem / out_dim3;
    const int w_o = rem % out_dim3;

    int h_start = h_o * stride - padding;
    int w_start = w_o * stride - padding;
    int d_start = d_o * stride - padding;

    scalar_t max_val = -FLT_MAX;
    for (int k_d = 0; k_d < kernel_size; ++k_d) {
        for (int k_h = 0; k_h < kernel_size; ++k_h) {
            for (int k_w = 0; k_w < kernel_size; ++k_w) {
                int d = d_start + k_d * dilation;
                int h = h_start + k_h * dilation;
                int w = w_start + k_w * dilation;
                if (d >= 0 && d < in_dim3 && h >= 0 && h < in_dim1 && w >= 0 && w < in_dim2) {
                    scalar_t val = input[(
                        c * in_dim1 + h) * in_dim2 * in_dim3 +
                        w * in_dim3 + d +
                        batch_size * channels * in_dim1 * in_dim2 * in_dim3 * 0]; // assuming batch first
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }
    }
    output[idx] = max_val;
}

torch::Tensor max_pool3d_cuda(torch::Tensor input,
                             int kernel_size,
                             int stride,
                             int padding,
                             int dilation,
                             bool ceil_mode) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_dim1 = input.size(2);
    const int in_dim2 = input.size(3);
    const int in_dim3 = input.size(4);

    int out_dim1 = (in_dim1 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_dim2 = (in_dim2 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_dim3 = (in_dim3 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    if (ceil_mode) {
        out_dim1 = (in_dim1 + 2 * padding - 1) / stride;
        out_dim2 = (in_dim2 + 2 * padding - 1) / stride;
        out_dim3 = (in_dim3 + 2 * padding - 1) / stride;
    }

    auto output = torch::empty({batch_size, channels, out_dim1, out_dim2, out_dim3}, input.options());

    const int threads = 256;
    const int elements = batch_size * channels * out_dim1 * out_dim2 * out_dim3;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_cuda", ([&]{
        max_pool3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, channels,
            in_dim1, in_dim2, in_dim3,
            out_dim1, out_dim2, out_dim3,
            kernel_size, stride, padding, dilation, ceil_mode);
    }));

    return output;
}
"""

maxpool3d_cpp_source = """
torch::Tensor max_pool3d_cuda(torch::Tensor input,
                             int kernel_size,
                             int stride,
                             int padding,
                             int dilation,
                             bool ceil_mode);
"""

max_pool3d = load_inline(
    name="max_pool3d",
    cpp_sources=maxpool3d_cpp_source,
    cuda_sources=maxpool3d_source,
    functions=["max_pool3d_cuda"],
    verbose=True,
    extra_cflags=["-D了多少宏?"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.max_pool3d = max_pool3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool3d.max_pool3d_cuda(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode
        )