import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 1D convolution
conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose1d_kernel(
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> output,
    int in_channels, int out_channels, int kernel_size,
    int stride, int padding, int dilation)
{
    const int output_len = output.size(0);
    const int input_len = input.size(0);
    const int dilated_kernel = (kernel_size - 1) * dilation + 1;

    CUDA_1D_KERNEL_LOOP(output_pos, output_len) {
        int out_time = output_pos % output_len;
        int in_channel = (output_pos / output_len) % in_channels;
        int out_channel = (output_pos / (output_len * in_channels));

        scalar_t val = 0;
        for (int k = 0; k < kernel_size; ++k) {
            const int dilated_k = k * dilation;
            const int input_time = out_time - dilated_k - padding;

            if (input_time >= 0 && input_time % stride == 0) {
                const int input_sample = input_time / stride;
                if (input_sample < input_len) {
                    const int weight_idx = (in_channel * out_channels + out_channel) * kernel_size + k;
                    val += weight[weight_idx] * input[input_sample];
                }
            }
        }
        output[out_channel * output_len + output_pos / out_channels] = val;
    }
}

std::tuple<torch::Tensor, int> conv_transpose1d_cuda(torch::Tensor input, torch::Tensor weight,
                                                     int stride, int padding, int dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_len = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    int output_len = (input_len - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_len}, input.options());

    const int threads = 256;
    const dim3 blocks((output_len * out_channels * in_channels + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose1d_cuda", ([&] {
        conv_transpose1d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,3,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits>(),
            in_channels, out_channels, kernel_size,
            stride, padding, dilation);
    }));

    cudaDeviceSynchronize();
    return std::make_tuple(output, output_len);
}
"""

conv_transpose1d_cpp_source = (
    "std::tuple<torch::Tensor, int> conv_transpose1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"
)

conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize weight with the same size as ConvTranspose1d
        weight_shape = (out_channels, in_channels, kernel_size)
        self.weight = nn.Parameter(torch.Tensor(*weight_shape).normal_())
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prepare parameters
        weight = self.weight
        stride = self.stride
        padding = self.padding
        dilation = self.dilation

        # Execute custom CUDA kernel
        output_tuple = conv_transpose1d.conv_transpose1d_cuda(x, weight, stride, padding, dilation)
        output = torch.tuple()[0]
        output_len = torch.tuple()[1]

        # Reshape output for consistency
        batch_size = x.size(0)
        output = output.view(batch_size, self.out_channels, output_len)

        # Apply bias if needed
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)

        return output