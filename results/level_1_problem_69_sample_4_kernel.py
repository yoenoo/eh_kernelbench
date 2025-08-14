import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(const scalar_t* input,
                                       const scalar_t* weight,
                                       scalar_t* output,
                                       int batch_size,
                                       int in_channels,
                                       int out_channels,
                                       int kernel_h,
                                       int kernel_w,
                                       int input_h,
                                       int input_w,
                                       int output_h,
                                       int output_w,
                                       int stride_h,
                                       int stride_w,
                                       int padding_h,
                                       int padding_w,
                                       int dilation_h,
                                       int dilation_w,
                                       int groups,
                                       bool bias) {
    // Calculate output coordinates
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (b >= batch_size || h_out >= output_h || w_out >= output_w) {
        return;
    }

    scalar_t val = 0;
    for (int k_h = 0; k_h < kernel_h; ++k_h) {
        for (int k_w = 0; k_w < kernel_w; ++k_w) {
            // Compute input coordinates
            int h_in = h_out * stride_h - padding_h - dilation_h * k_h;
            int w_in = w_out * stride_w - padding_w - dilation_w * k_w;

            // Check if input coordinates are valid
            if (h_in < 0 || h_in >= input_h || w_in < 0 || w_in >= input_w) {
                continue;
            }

            for (int c_in = 0; c_in < in_channels; ++c_in) {
                for (int c_out = 0; c_out < out_channels; ++c_out) {
                    // Get weight index considering kernel dimensions and groups
                    int weight_idx = (c_out * in_channels + c_in) * kernel_h * kernel_w + k_h * kernel_w + k_w;
                    // Accumulate the result
                    val += input[b * in_channels * input_h * input_w + c_in * input_h * input_w + h_in * input_w + w_in] *
                           weight[weight_idx];
                }
            }
        }
    }
    output[b * out_channels * output_h * output_w + c_out * output_h * output_w + h_out * output_w + w_out] = val;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int kernel_h,
                                   int kernel_w,
                                   int stride_h,
                                   int stride_w,
                                   int padding_h,
                                   int padding_w,
                                   int dilation_h,
                                   int dilation_w,
                                   int groups,
                                   bool bias) {
    // Calculate output dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_h = input.size(2);
    auto input_w = input.size(3);

    int output_h = (input_h - 1) * stride_h - 2 * padding_h + kernel_h + dilation_h * (kernel_h - 1) + 1;
    int output_w = (input_w - 1) * stride_w - 2 * padding_w + kernel_w + dilation_w * (kernel_w - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());

    dim3 threads(16, 16);
    dim3 blocks((output_w + threads.x - 1) / threads.x,
                (output_h + threads.y - 1) / threads.y,
                batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            input_h,
            input_w,
            output_h,
            output_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            bias);
    }));

    return output;
}
"""

conv_transpose_cpp_source = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int kernel_h,
                                   int kernel_w,
                                   int stride_h,
                                   int stride_w,
                                   int padding_h,
                                   int padding_w,
                                   int dilation_h,
                                   int dilation_w,
                                   int groups,
                                   bool bias);
"""

conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights similar to PyTorch's ConvTranspose2d
        kernel_h, kernel_w = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_h, kernel_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation
        
        # Ensure all inputs are on the same device as the weights
        x = x.to(self.weight.device)
        
        return conv_transpose.conv_transpose2d_cuda(
            x,
            self.weight,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            self.groups,
            self.bias is not None
        )