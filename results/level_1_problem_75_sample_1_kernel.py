import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size, int input_channels, int input_height, int input_width,
    int output_channels, int output_height, int output_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups) {

    CUDA_1D_KERNEL_LOOP(output_idx, batch_size * output_channels * output_height * output_width) {
        int w_out = output_idx % output_width;
        int h_out = (output_idx / output_width) % output_height;
        int c_out_group = (output_idx / (output_width * output_height)) % (output_channels / groups);
        int group_idx = (output_idx / (output_channels * output_height * output_width)) % groups;
        int n = output_idx / (output_channels * output_height * output_width);

        int c_in = group_idx * (input_channels / groups) + c_out_group;

        scalar_t val = 0;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Compute input coordinates
                int h_in = (h_out * stride_h - padding_h - kh * dilation_h);
                int w_in = (w_out * stride_w - padding_w - kw * dilation_w);

                // Check input boundaries
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    int weight_offset = (group_idx * output_channels/(groups) + c_out_group) * kernel_h * kernel_w + kh * kernel_w + kw;
                    int input_offset = ((n * input_channels + c_in) * input_height + h_in) * input_width + w_in;
                    val += input[input_offset] * weight[weight_offset];
                }
            }
        }
        int output_offset = ((n * output_channels + group_idx * (output_channels/groups) + c_out_group) * output_height + h_out) * output_width + w_out;
        output[output_offset] = val;
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w,
                                   int padding_h, int padding_w, int dilation_h, int dilation_w, int groups) {

    const int batch_size = input.size(0);
    const int input_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_channels = weight.size(0) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    // Compute output dimensions
    const int output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1;
    const int output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1;

    auto output = torch::zeros({batch_size, output_channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int elements = batch_size * output_channels * output_height * output_width;
    const int blocks = (elements + threads - 1) / threads;

    conv_transpose2d_kernel<float><<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, input_channels, input_height, input_width,
        output_channels, output_height, output_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups);

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w,
                                   int padding_h, int padding_w, int dilation_h, int dilation_w, int groups);
"""

conv_transposed_op = load_inline(
    name="conv_transposed_op",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Initialize weights like ConvTranspose2d (note: bias not implemented here)
        weight_shape = (in_channels, out_channels // groups, *kernel_size)
        self.weight = nn.Parameter(torch.randn(*weight_shape))
        # Transpose for deconv (swap in and out channels)
        self.weight.data = self.weight.data.transpose(0, 1).contiguous()

    def forward(self, x):
        return conv_transposed_op.conv_transpose2d_cuda(
            x, self.weight, self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1], self.groups
        )