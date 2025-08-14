import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 1D Transposed Convolution
conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Transposed Convolution (Deconvolution) in 1D
// Based on the principles of upsampling followed by convolution
template <typename scalar_t>
__global__ void transposed_conv1d_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

    // Calculate output spatial dimension
    const int input_length = input.size(1);
    const int output_length = (input_length - 1) * stride + 1 - 2 * padding + (kernel_size - 1)*dilation + 2*padding;

    // Thread indices
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * out_channels * output_length) return;

    // Compute spatial and channel indices
    int batch = tid / (out_channels * output_length);
    int oc = (tid / output_length) % out_channels;
    int l = tid % output_length;

    // Iterate over input channels and kernel
    scalar_t val = 0.0;
    for (int ic = 0; ic < in_channels; ic++) {
        for (int k = 0; k < kernel_size; k++) {
            // Compute input position
            int il = l - padding - k*dilation;
            if (il % stride == 0 && il >= 0 && il < (input_length - 1)*stride + (kernel_size - 1)*dilation + 1) {
                int il_input = il / stride;
                if (il_input < 0 || il_input >= input_length) continue;
                val += weight[oc][ic][k] * input[batch * in_channels + ic][il_input];
            }
        }
    }

    output[batch * out_channels + oc][l] = val;
}

std::vector<torch::Tensor> transposed_conv1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    int output_length = (input.size(2) - 1)*stride + 1;
    output_length += (kernel_size - 1)*dilation + 2*padding;

    torch::Tensor output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    const int threads = 256;
    int elements = batch_size * out_channels * output_length;
    int blocks = (elements + threads - 1) / threads;

    const int block_size = 256;
    const int num_blocks = (elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "transposed_conv1d_cuda", ([&]{
        transposed_conv1d_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,3,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits>(),
            batch_size, in_channels, out_channels,
            kernel_size, stride, padding, dilation);
    }));

    return {output};
}
"""

# Compile the inline CUDA kernel
conv_transpose_cuda = load_inline(
    name="transposed_conv",
    cpp_sources="",
    cuda_sources=conv_transpose1d_source,
    functions=["transposed_conv1d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # Initialize weights similar to PyTorch's ConvTranspose1d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        outputs = conv_transpose_cuda.transposed_conv1d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.dilation
        )[0]
        if self.bias is not None:
            outputs += self.bias.view(1, -1, 1)
        return outputs