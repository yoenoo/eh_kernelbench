import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom ConvTranspose1d CUDA kernel implementation
conv_transpose_1d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                       \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

template <typename scalar_t>
__global__ void conv_transpose1d_kernel(const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ weight,
                                       scalar_t* output,
                                       int batch_size,
                                       int in_channels,
                                       int out_channels,
                                       int input_length,
                                       int output_length,
                                       int kernel_size,
                                       int stride,
                                       int padding,
                                       int dilation) {

    CUDA_1D_KERNEL_LOOP(index, batch_size * out_channels * output_length) {
        int output_pos = index % output_length;
        int channel_out = (index / output_length) % out_channels;
        int n = index / (out_channels * output_length);

        scalar_t val = 0;
        for (int kernel_idx = 0; kernel_idx < kernel_size; kernel_idx++) {
            int input_idx = (output_pos + padding - dilation * kernel_idx);
            if (input_idx < 0 || input_idx >= input_length)
                continue;
            for (int channel_in = 0; channel_in < in_channels; channel_in++) {
                val += input[n * in_channels * input_length + channel_in * input_length + input_idx] *
                       weight[channel_out * in_channels * kernel_size + channel_in * kernel_size + kernel_idx];
            }
        }
        output[index] = val;
    }
}

torch::Tensor conv_transpose1d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int stride,
                                   int padding,
                                   int dilation,
                                   int kernel_size) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_length = input.size(2);
    const auto out_channels = weight.size(0);
    const auto output_length = (input_length - 1) * stride + dilation * (kernel_size - 1) + 1 - 2 * padding;

    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    dim3 blocks(
        (batch_size * out_channels * output_length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose1d_cuda", ([&] {
        conv_transpose1d_kernel<scalar_t><<<blocks, threads, 0, at::cuda::current_stream()>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_length,
            output_length,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_cuda, "1D Transposed Convolution Forward");
}
"""

conv_transpose_1d_cpp = """
#include <torch/extension.h>
torch::Tensor conv_transpose1d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int stride,
                                   int padding,
                                   int dilation,
                                   int kernel_size);
"""

custom_conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources=conv_transpose_1d_cpp,
    cuda_sources=conv_transpose_1d_source,
    functions="conv_transpose1d_cuda",
    verbose=True,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        # Initialize weights similar to PyTorch's ConvTranspose1d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = custom_conv_transpose.conv_transpose1d_cuda(
            x.cuda(),
            self.weight.cuda(),
            self.stride,
            self.padding,
            self.dilation,
            self.kernel_size
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
        return output

# Initialization remains same as original but using new parameters
# Note: get_init_inputs would now need to pass parameters differently but as per problem statement, we are only modifying ModelNew