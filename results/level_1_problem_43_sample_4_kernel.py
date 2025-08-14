import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

maxpool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void custom_maxpool3d_kernel(const scalar_t* input, scalar_t* output,
    int batch_size, int channels, int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int kernel_size, int stride, int padding, int dilation,
    int d_kernel, int h_kernel, int w_kernel) {

    const int batch_idx = blockIdx.x;
    const int channel = blockIdx.y;
    const int out_d = blockIdx.z;
    const int out_h = threadIdx.z;
    const int out_w = threadIdx.y * blockDim.x + threadIdx.x;

    if (out_h >= out_height || out_w >= out_width) return;

    const int in_offset = (batch_idx * channels + channel) *
        in_depth * in_height * in_width;
    const int out_offset = (batch_idx * channels + channel) *
        out_depth * out_height * out_width;

    const int d_start = out_d * stride - padding;
    const int h_start = out_h * stride - padding;
    const int w_start = out_w * stride - padding;

    scalar_t max_val = -FLT_MAX;
    for (int kd = 0; kd < d_kernel; ++kd) {
        const int d = d_start + kd * dilation;
        if (d < 0 || d >= in_depth) continue;
        for (int kh = 0; kh < h_kernel; ++kh) {
            const int h = h_start + kh * dilation;
            if (h < 0 || h >= in_height) continue;
            for (int kw = 0; kw < w_kernel; ++kw) {
                const int w = w_start + kw * dilation;
                if (w < 0 || w >= in_width) continue;
                const auto val = input[in_offset + d * in_height * in_width +
                                      h * in_width + w];
                if (val > max_val) max_val = val;
            }
        }
    }

    const int out_index = out_offset + out_d * out_height * out_width +
        out_h * out_width + out_w;
    output[out_index] = max_val;
}

torch::Tensor custom_maxpool3d_forward(torch::Tensor input,
    int kernel_size, int stride, int padding, int dilation,
    int d_kernel, int h_kernel, int w_kernel,
    int out_depth, int out_height, int out_width) {

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto in_depth = input.size(2);
    const auto in_height = input.size(3);
    const auto in_width = input.size(4);

    auto output = torch::empty({batch_size, channels, out_depth,
                               out_height, out_width}, input.options());

    const int block_dim = 256;
    dim3 threads(16, 16, 1); // XYZ threading
    threads.z = (out_height + threads.y - 1) / threads.y;
    threads.y = (out_width + threads.x - 1) / threads.x;
    threads.x = 16; // Adjust based on sm count

    dim3 blocks(batch_size, channels, out_depth);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool3d_cuda", ([&] {
        custom_maxpool3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, channels, in_depth, in_height, in_width,
            out_depth, out_height, out_width,
            kernel_size, stride, padding, dilation,
            d_kernel, h_kernel, w_kernel);
    }));

    return output;
}
"""

cpp_source = """
torch::Tensor custom_maxpool3d_forward(torch::Tensor input,
    int kernel_size, int stride, int padding, int dilation,
    int d_kernel, int h_kernel, int w_kernel,
    int out_depth, int out_height, int out_width);
"""

maxpool3d_cuda = load_inline(name='maxpool3d_cuda',
                            cuda_sources=maxpool3d_source,
                            cpp_sources=cpp_source,
                            functions=['custom_maxpool3d_forward'],
                            verbose=True)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0,
                 dilation: int = 1, return_indices: bool = False,
                 ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.d_kernel = self.h_kernel = self.w_kernel = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output dimensions
        N, C, D, H, W = x.size()
        kernel_d = kernel_h = kernel_w = self.kernel_size
        stride_d = stride_h = stride_w = self.stride

        # Calculate output dimensions using same logic as PyTorch
        output_spec = lambda i, k, s, p, dilation, ceil: (
            (i + 2*p - dilation*(k-1) - 1) // s + 1
        )
        out_depth = output_spec(D, kernel_d, stride_d, self.padding, self.dilation, self.ceil_mode)
        out_height = output_spec(H, kernel_h, stride_h, self.padding, self.dilation, self.ceil_mode)
        out_width = output_spec(W, kernel_w, stride_w, self.padding, self.dilation, self.ceil_mode)

        # Launch CUDA kernel
        return maxpool3d_cuda.custom_maxpool3d_forward(
            x.cuda(),
            self.kernel_size, self.stride, self.padding, self.dilation,
            self.d_kernel, self.h_kernel, self.w_kernel,
            out_depth, out_height, out_width
        ).cuda()