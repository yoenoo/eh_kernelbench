import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

kernel_size = 3
stride = 2
padding = 1

# Custom CUDA kernel for 3D average pooling
avg_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void avg_pool3d_kernel(const scalar_t* input, scalar_t* output,
        const int batch_size, const int channels,
        const int in_depth, const int in_height, const int in_width,
        const int out_depth, const int out_height, const int out_width,
        const int kernel_size, const int stride, const int padding) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * out_depth * out_height * out_width) return;

    const int w = idx % out_width;
    const int h = (idx / out_width) % out_height;
    const int d = (idx / (out_width * out_height)) % out_depth;
    const int c = (idx / (out_width * out_height * out_depth)) % channels;
    const int n = idx / (out_width * out_height * out_depth * channels);

    const int in_d_start = d * stride - padding;
    const int in_h_start = h * stride - padding;
    const int in_w_start = w * stride - padding;

    scalar_t sum = 0;
    int count = 0;

    for (int kd = 0; kd < kernel_size; ++kd) {
        const int in_d = in_d_start + kd;
        if (in_d < 0 || in_d >= in_depth) continue;
        for (int kh = 0; kh < kernel_size; ++kh) {
            const int in_h = in_h_start + kh;
            if (in_h < 0 || in_h >= in_height) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int in_w = in_w_start + kw;
                if (in_w < 0 || in_w >= in_width) continue;
                const int offset = n * channels * in_depth * in_height * in_width +
                                c * in_depth * in_height * in_width +
                                in_d * in_height * in_width +
                                in_h * in_width + in_w;
                sum += input[offset];
                count++;
            }
        }
    }
    output[idx] = sum / static_cast<scalar_t>(count);
}

std::vector<int64_t> output_shape(int64_t in_depth, int64_t in_height, int64_t in_width, int64_t kernel_size, int64_t stride, int64_t padding) {
    auto output_depth = (in_depth + 2 * padding - kernel_size) / stride + 1;
    auto output_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    auto output_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    return {output_depth, output_height, output_width};
}

torch::Tensor avg_pool3d_cuda(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding) {
    const auto in_depth = input.size(2);
    const auto in_height = input.size(3);
    const auto in_width = input.size(4);

    auto out_shape = output_shape(in_depth, in_height, in_width, kernel_size, stride, padding);
    auto output = torch::empty({input.size(0), input.size(1), out_shape[0], out_shape[1], out_shape[2]}, input.options());

    int batch_size = input.size(0);
    int channels = input.size(1);

    const dim3 threads(256);
    const dim3 blocks((batch_size * channels * out_shape[0] * out_shape[1] * out_shape[2] + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool3d_cuda", ([&] {
        avg_pool3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, channels,
            in_depth, in_height, in_width,
            out_shape[0], out_shape[1], out_shape[2],
            kernel_size, stride, padding);
    }));

    return output;
}
"""

avg_pool3d_cpp_source = "torch::Tensor avg_pool3d_cuda(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding);"

avg_pool3d = load_inline(
    name="avg_pool3d",
    cpp_sources=avg_pool3d_cpp_source,
    cuda_sources=avg_pool3d_source,
    functions=["avg_pool3d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avg_pool3d.avg_pool3d_cuda(x, self.kernel_size, self.stride, self.padding)