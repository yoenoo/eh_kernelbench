import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void ConvTranspose2dKernel(const scalar_t* input, const scalar_t* weight, scalar_t* output,
    int batch_size, int in_channels, int out_channels, int kernel_h, int kernel_w,
    int input_h, int input_w, int output_h, int output_w,
    int stride_h, int stride_w, int padding_h, int padding_w) {

    const int channels_out = out_channels;
    const int channels_in = in_channels;

    CUDA_1D_KERNEL_LOOP(output_idx, batch_size * channels_out * output_h * output_w) {
        int w_out = output_idx % output_w;
        int h_out = (output_idx / output_w) % output_h;
        int c_out = (output_idx / (output_w * output_h)) % channels_out;
        int n = output_idx / (channels_out * output_h * output_w);

        scalar_t val = 0;
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                // Compute input coordinates
                int h_in = h_out * stride_h - padding_h + kernel_row;
                int w_in = w_out * stride_w - padding_w + kernel_col;
                if (h_in < 0 || h_in >= input_h || w_in < 0 || w_in >= input_w) {
                    continue;
                }
                for (int c_in = 0; c_in < channels_in; ++c_in) {
                    val += weight[c_out * channels_in * kernel_h * kernel_w +
                                c_in * kernel_h * kernel_w +
                                kernel_row * kernel_w + kernel_col] *
                            input[n * channels_in * input_h * input_w +
                                c_in * input_h * input_w +
                                h_in * input_w + w_in];
                }
            }
        }
        output[output_idx] = val;
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_h = input.size(2);
    const auto input_w = input.size(3);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);
    const auto out_channels = weight.size(0);

    const int output_h = (input_h - 1) * stride_h - 2 * padding_h + kernel_h;
    const int output_w = (input_w - 1) * stride_w - 2 * padding_w + kernel_w;

    auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());

    const int threads = 1024;
    int blocks = (batch_size * out_channels * output_h * output_w + threads - 1) / threads;

    const int kernel_type = input.scalar_type();
    if (kernel_type == torch::kFloat32) {
        ConvTranspose2dKernel<float><<<blocks, threads>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
            batch_size, in_channels, out_channels,
            kernel_h, kernel_w, input_h, input_w, output_h, output_w,
            stride_h, stride_w, padding_h, padding_w
        );
    }
    else {
        AT_ERROR("Unsupported tensor type");
    }

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w);
"""

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        # PyTorch's ConvTranspose2d has weight of shape (in_channels, out_channels, kernel_h, kernel_w)
        # However, our custom kernel expects weight to be (out_channels, in_channels, kernel_h, kernel_w)
        # So transpose the weight dimensions here
        weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.weight = weight
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        out = conv_transpose2d.conv_transpose2d_cuda(x, self.weight, self.stride[0], self.stride[1], self.padding[0], self.padding[1])
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out