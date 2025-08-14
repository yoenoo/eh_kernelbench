import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int batch_size, int in_channels, int height_in, int width_in,
    int kernel_size, int stride, int padding) {

    const int B = blockIdx.z;
    const int C = blockIdx.y;
    const int out_h = blockIdx.x;
    const int out_w = threadIdx.x;

    scalar_t sum = 0;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int h_in = -padding + out_h*stride + kh;
            const int w_in = -padding + out_w*stride + kw;
            if (h_in >=0 && h_in < height_in && w_in >=0 && w_in < width_in) {
                sum += input[B][C][h_in][w_in] * weight[C][0][kh][kw];
            }
        }
    }
    output[B][C][out_h][out_w] = sum;
}

std::vector<int64_t> compute_output_shape(int64_t height_in, int64_t width_in, int64_t kernel_size, int64_t stride, int64_t padding) {
    int64_t height_out = (height_in + 2*padding - kernel_size) / stride + 1;
    int64_t width_out = (width_in + 2*padding - kernel_size) / stride + 1;
    return {height_out, width_out};
}

torch::Tensor depthwise_conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size,
    int stride,
    int padding) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto height_in = input.size(2);
    const auto width_in = input.size(3);

    auto output_shape = compute_output_shape(height_in, width_in, kernel_size, stride, padding);
    auto output = torch::zeros({batch_size, in_channels, output_shape[0], output_shape[1]}, input.options());

    const dim3 threads(32, 1, 1); // Changed to 32 threads per block
    dim3 blocks(output_shape[1], in_channels, batch_size);

    const int shared_mem_size = 0;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_forward_cuda", ([&]{
        depthwise_conv2d_forward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, in_channels, height_in, width_in,
            kernel_size, stride, padding);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor depthwise_conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size,
    int stride,
    int padding);
"""

depthwise_conv2d_cuda = load_inline(
    name='depthwise_conv2d_cuda',
    cpp_sources=cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=['depthwise_conv2d_forward_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        assert in_channels == out_channels, "Depthwise requires in_channels == out_channels"
        self.weight = nn.Parameter(torch.empty((in_channels, 1, kernel_size, kernel_size)))
        self.bias = bias
        if bias:
            self.bias_term = nn.Parameter(torch.empty(in_channels))
        else:
            self.bias_term = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_term is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_term, -bound, bound)

    def forward(self, x):
        output = depthwise_conv2d_cuda.depthwise_conv2d_forward_cuda(
            x, self.weight, self.kernel_size, self.stride, self.padding)
        if self.bias_term is not None:
            output = output + self.bias_term.view(1, -1, 1, 1)
        return output