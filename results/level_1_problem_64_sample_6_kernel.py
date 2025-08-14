import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose1d
conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups) {

    CUDA_KERNEL_LOOP(output_index, batch_size * out_channels * output_length) {
        int batch = output_index / (out_channels * output_length);
        int channel = (output_index / output_length) % out_channels;
        int output_pos = output_index % output_length;

        scalar_t sum = 0;

        // Iterate over input channels and kernel positions
        for (int k = 0; k < kernel_size; ++k) {
            // Determine the input position
            int input_pos = (output_pos - k - output_padding) / stride;

            // Check if input_pos is within bounds and the effective input
            // (using reversed kernel for transpose)
            if ((output_pos - k - output_padding) % stride == 0 && 
                input_pos >= -padding && input_pos < input_length) {
                // Compute the input channel and input's effective channel due to groups
                int in_channel_group = channel / (out_channels / groups);
                int group_in_channels = in_channels / groups;

                // Iterate over groups
                for (int g = 0; g < group_in_channels; ++g) {
                    int in_channel = in_channel_group * group_in_channels + g;
                    // Get input value at (batch, in_channel, input_pos)
                    int input_offset = batch * in_channels * input_length + 
                                      in_channel * input_length + 
                                      (input_pos + padding);  // Adding padding?

                    // Get weight for the reverse kernel
                    // weight is [in_channels, out_channels_per_group, kernel_size]
                    int out_per_group = out_channels / groups;
                    int weight_offset = in_channel * out_per_group * kernel_size + 
                                       (channel % out_per_group) * kernel_size + 
                                       k;

                    sum += input[input_offset] * weight[weight_offset];
                }
            }
        }

        output[output_index] = sum;
    }
}

torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding,
    int groups) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(1) * groups;

    // Compute output_length according to ConvTranspose1d formula
    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    dim3 blocks((output.numel() + 512 - 1) / 512);
    dim3 threads(512);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose1d_cuda", ([&] {
        conv_transpose1d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups);
    }));

    return output;
}
"""

conv_transpose1d_cpp = "torch::Tensor conv_transpose1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int output_padding, int groups);"

conv_transpose_op = load_inline(
    name="conv_transpose1d_op",
    cpp_sources=conv_transpose1d_cpp,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        # Initialize weights like PyTorch's ConvTranspose1d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # If bias is required, handle here (not included in current kernel)
        # For simplicity, bias is omitted as per the original model (bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose_op.conv_transpose1d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )