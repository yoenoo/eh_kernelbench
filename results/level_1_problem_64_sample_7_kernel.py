import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose1d
conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose1d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits> output,
    int batch_size, int input_channels, int output_channels, int input_length,
    int kernel_size, int stride, int padding, int output_padding) {

    CUDA_KERNEL_LOOP(index, batch_size * output_channels * (input_length * stride + output_padding)) {
        int output_idx = index;
        int b = output_idx / (output_channels * (input_length * stride + output_padding));
        int rest = output_idx % (output_channels * (input_length * stride + output_padding));
        int oc = rest / (input_length * stride + output_padding);
        int l = rest % (input_length * stride + output_padding);

        scalar_t val = 0;
        for (int ic = 0; ic < input_channels; ic++) {
            for (int k = 0; k < kernel_size; k++) {
                int input_l = (l - k - output_padding) / stride + padding;
                if ((l - k - output_padding) % stride == 0 && input_l >= 0 && input_l < input_length) {
                    int w_index = oc * input_channels * kernel_size + ic * kernel_size + k;
                    val += input[b][ic][0][input_l] * weight[oc][ic][0][k];
                }
            }
        }
        output[b][oc][0][l] = val;
    }
}

torch::Tensor conv_transpose1d_forward(torch::Tensor input, torch::Tensor weight,
        int stride, int padding, int output_padding, int kernel_size) {

    const auto batch_size = input.size(0);
    const auto input_channels = input.size(1);
    const auto output_channels = weight.size(0);
    const auto input_length = input.size(2);
    const auto output_length = input_length * stride + 2 * padding - kernel_size + 1 + output_padding;

    auto output = torch::zeros({batch_size, output_channels, 1, output_length}, input.options());

    const int threads = 256;
    const int output_elements = batch_size * output_channels * output_length;
    const int blocks = (output_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose1d_forward", ([&] {
        conv_transpose1d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits>(),
            batch_size, input_channels, output_channels, input_length,
            kernel_size, stride, padding, output_padding);
    }));

    return output.view({batch_size, output_channels, output_length});
}

"""

conv_transpose1d_cpp_source = (
    "torch::Tensor conv_transpose1d_forward(torch::Tensor input, torch::Tensor weight,"
    "int stride, int padding, int output_padding, int kernel_size);"
)

# Compile the inline CUDA code for ConvTranspose1d
conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_forward"],
    verbose=False,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        # Adjust weight shape to match custom kernel requirements (out_channels, in_channels, 1, kernel_size)
        weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, 1, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        self.register_parameter('weight', weight)
        self.conv_transpose1d_forward = conv_transpose1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input to 4D (N, C, 1, L) for kernel compatibility
        x_reshaped = x.unsqueeze(2)
        output = self.conv_transpose1d_forward.conv_transpose1d_forward(
            x_reshaped, self.weight, self.stride, self.padding, self.output_padding, self.weight.size(3)
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output.squeeze(2)