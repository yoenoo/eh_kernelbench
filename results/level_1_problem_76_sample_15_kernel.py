import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel code for optimized 1D convolution
conv1d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void optimized_conv1d_forward(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_length,
    int kernel_size,
    int stride,
    int dilation,
    int padding_l,
    int out_length) {

    const int output_size = batch_size * out_channels * out_length;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    int batch_idx = idx / (out_channels * out_length);
    int oc = (idx / out_length) % out_channels;
    int t = idx % out_length;

    scalar_t sum = 0;
    for (int kk = 0; kk < kernel_size; ++kk) {
        int in_t = t * stride + kk * dilation - padding_l;
        if (in_t < 0 || in_t >= in_length) continue;

        for (int ic = 0; ic < in_channels; ++ic) {
            sum += weight[oc * in_channels * kernel_size + ic * kernel_size + kk] *
                   input[batch_idx * in_channels * in_length + ic * in_length + in_t];
        }
    }
    output[idx] = sum;
}

torch::Tensor optimized_conv1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int dilation, int padding_l, int out_length) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_length = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({batch_size, out_channels, out_length}, output_options);

    const int threads = 256;
    const int blocks = (batch_size * out_channels * out_length + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "optimized_conv1d_forward", ([&] {
        optimized_conv1d_forward<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            in_length,
            kernel_size,
            stride,
            dilation,
            padding_l,
            out_length);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv1d_cpp_source = (
    "torch::Tensor optimized_conv1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int dilation, int padding_l, int out_length);"
)

# Compile the custom CUDA operator
custom_conv1d = load_inline(
    name="custom_conv1d",
    cpp_sources=conv1d_cpp_source,
    cuda_sources=conv1d_source,
    functions=["optimized_conv1d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.dilation = dilation
        self.padding_l = 0  # Assuming no padding, adjust if needed
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        # Initialize weights (simplified; use proper initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None
        # Precompute output length for forward
        self.out_length = self.compute_out_length(in_channels, kernel_size, stride, dilation)

    def compute_out_length(self, in_channels, kernel_size, stride, dilation):
        # For simplicity, assuming no padding and input length provided via input tensor
        # This would need proper parameterization based on input size (e.g., at forward)
        # Here placeholder logic, adjust according to actual input size
        length = 524280  # Placeholder value (from the provided problem's get_inputs)
        return (length + 2 * self.padding_l - dilation * (kernel_size - 1) - 1) // stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamically compute output length from input
        input_length = x.size(2)
        out_length = (input_length + 2 * self.padding_l - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        output = custom_conv1d.optimized_conv1d_cuda(
            x,
            self.weight,
            self.stride,
            self.dilation,
            self.padding_l,
            out_length
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
        return output