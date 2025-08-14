import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for MaxPool3d
max_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool3d_kernel(
    const torch::PackedTensorAccessor<scalar_t,5> input,
    torch::PackedTensorAccessor<scalar_t,5> output,
    int64_t batch_size, int64_t channels,
    int64_t input_dim1, int64_t input_dim2, int64_t input_dim3,
    int64_t kernel_size, int64_t stride,
    int64_t padding, int64_t dilation,
    int64_t output_dim1, int64_t output_dim2, int64_t output_dim3,
    bool ceil_mode) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_dim1 * output_dim2 * output_dim3) return;

    int batch = idx / (channels * output_dim1 * output_dim2 * output_dim3);
    int c = (idx % (channels * output_dim1 * output_dim2 * output_dim3)) / (output_dim1 * output_dim2 * output_dim3);
    int out_d = (idx % (output_dim1 * output_dim2 * output_dim3)) / (output_dim2 * output_dim3);
    int out_h = (idx % (output_dim2 * output_dim3)) / output_dim3;
    int out_w = idx % output_dim3;

    int d_start = out_d * stride - padding;
    int h_start = out_h * stride - padding;
    int w_start = out_w * stride - padding;

    scalar_t max_val = -FLT_MAX;
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int d = d_start + dilation * kd;
                int h = h_start + dilation * kh;
                int w = w_start + dilation * kw;
                
                if (d < 0 || h < 0 || w < 0 ||
                    d >= input_dim1 || h >= input_dim2 || w >= input_dim3) continue;
                
                scalar_t val = input[batch][c][d][h][w];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    output[batch][c][out_d][out_h][out_w] = max_val;
}

std::tuple<torch::Tensor> max_pool3d_cuda(
    torch::Tensor input,
    int64_t kernel_size, int64_t stride,
    int64_t padding, int64_t dilation,
    bool ceil_mode) {

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_dim1 = input.size(2);
    const auto input_dim2 = input.size(3);
    const auto input_dim3 = input.size(4);

    // Calculate output dimensions
    auto output_dim1 = (input_dim1 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto output_dim2 = (input_dim2 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto output_dim3 = (input_dim3 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    if (ceil_mode) {
        output_dim1 += ((input_dim1 + 2 * padding - dilation * (kernel_size - 1)) % stride != 0);
        output_dim2 += ((input_dim2 + 2 * padding - dilation * (kernel_size - 1)) % stride != 0);
        output_dim3 += ((input_dim3 + 2 * padding - dilation * (kernel_size - 1)) % stride != 0);
    }

    torch::Tensor output = torch::empty({batch_size, channels, output_dim1, output_dim2, output_dim3}, input.options());

    int num_threads = 256;
    int num_blocks = (batch_size * channels * output_dim1 * output_dim2 * output_dim3 + num_threads - 1) / num_threads;

    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool3d_cuda", ([&]{
        max_pool3d_kernel<scalar_t><<<num_blocks, num_threads, 0, stream>>>(
            input.packed_accessor<scalar_t,5>(),
            output.packed_accessor<scalar_t,5>(),
            batch_size, channels,
            input_dim1, input_dim2, input_dim3,
            kernel_size, stride,
            padding, dilation,
            output_dim1, output_dim2, output_dim3,
            ceil_mode);
    }));

    return output;
}
"""

# Compile the CUDA kernel
max_pool3d = load_inline(
    name='max_pool3d',
    cpp_sources="",
    cuda_sources=max_pool3d_source,
    functions=['max_pool3d_cuda'],
    verbose=True,
    extra_cflags=['-DWITH_CUDA'],
    extra_ldflags=['']
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
        self.max_pool3d_cuda = max_pool3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.return_indices:
            raise NotImplementedError("Custom kernel currently does not support return_indices")
        
        # Assuming input is on CUDA device
        x = x.cuda()
        
        output = self.max_pool3d_cuda.max_pool3d_cuda(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode
        )
        
        return output

def get_inputs():
    x = torch.rand(batch_size, channels, dim1, dim2, dim3).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]