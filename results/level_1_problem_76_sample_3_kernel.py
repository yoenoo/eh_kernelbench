import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom 1D convolution CUDA kernel with dilation and stride
conv1d_kernel = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

template <typename scalar_t>
__global__ void conv1d_forward_kernel(const scalar_t* __restrict__ input,
                                     const scalar_t* __restrict__ weight,
                                     scalar_t* __restrict__ output,
                                     const int batch_size,
                                     const int in_channels,
                                     const int out_channels,
                                     const int input_length,
                                     const int kernel_size,
                                     const int output_length,
                                     const int stride,
                                     const int dilation) {

    // Each block handles one output feature map (out_channel) and one output position (output_pos)
    const int out_channel = blockIdx.z;
    const int output_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_pos >= output_length)
        return;

    const int in_channel = blockIdx.y;
    const int batch_idx = blockIdx.z / out_channels; // Assumes out_channels <= max blocks per grid

    const int input_start = output_pos * stride;
    const int input_end = input_start + dilation * (kernel_size - 1) + 1;

    scalar_t sum = 0;
    for (int k = 0; k < kernel_size; ++k) {
        int input_pos = input_start + dilation * k;
        if (input_pos < 0 || input_pos >= input_length)
            continue;
        sum += input[batch_idx * in_channels * input_length + in_channel * input_length + input_pos] *
               weight[out_channel * in_channels * kernel_size + in_channel * kernel_size + k];
    }

    // Accumulate contributions across input channels
    __shared__ scalar_t shared_sum[1024]; // Adjust size based on maximum output_length
    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    // Sum across the kernel elements and input channels using parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[batch_idx * out_channels * output_length + out_channel * output_length + output_pos] = shared_sum[0];
    }
}

std::tuple<torch::Tensor> custom_conv1d_forward(torch::Tensor input,
                                               torch::Tensor weight,
                                               int stride,
                                               int dilation) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_length = input.size(2);
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);

    // Compute output length
    const auto output_length = (input_length - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    const int threads = 256;
    dim3 blocks(output_length, in_channels, out_channels * batch_size);
    dim3 threadsPerBlock(threads);

    AT_DISPATCH_ALL_TYPES(input.type(), "conv1d_forward", ([&] {
        conv1d_forward_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_length,
            kernel_size,
            output_length,
            stride,
            dilation);
    }));

    return output;
}
"""

cpp_source = "std::tuple<torch::Tensor> custom_conv1d_forward(torch::Tensor input, torch::Tensor weight, int stride, int dilation);"

module = load_inline(
    name="custom_conv1d",
    cpp_sources=cpp_source,
    cuda_sources=conv1d_kernel,
    functions=["custom_conv1d_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = module.custom_conv1d_forward(x, self.weight, self.stride, self.dilation)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        return output