import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void ConvTranspose2dKernel(const scalar_t* __restrict__ input,
                                     const scalar_t* __restrict__ weight,
                                     scalar_t* __restrict__ output,
                                     int batch_size, int in_channels,
                                     int out_channels, int kernel_h, int kernel_w,
                                     int input_h, int input_w, int output_h, int output_w,
                                     int stride, int padding_h, int padding_w,
                                     int output_padding_h, int output_padding_w) {
    int batch = blockIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.z * blockDim.x + threadIdx.x;

    if (out_y >= output_h || out_x >= output_w) return;

    for (int c_out = threadIdx.z; c_out < out_channels; c_out += blockDim.z) {
        scalar_t sum = 0;
        for (int k_h = 0; k_h < kernel_h; ++k_h) {
            for (int k_w = 0; k_w < kernel_w; ++k_w) {
                int in_h = (out_y - k_h + padding_h - output_padding_h) / stride;
                int in_w = (out_x - k_w + padding_w - output_padding_w) / stride;
                if ((out_y - k_h + padding_h - output_padding_h) % stride != 0 ||
                    (out_x - k_w + padding_w - output_padding_w) % stride != 0) {
                    continue;
                }
                in_h = (out_y - k_h) / stride;
                in_w = (out_x - k_w) / stride;
                if (in_h < 0 || in_h >= input_h || in_w < 0 || in_w >= input_w) {
                    continue;
                }
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    sum += input[batch * in_channels * input_h * input_w +
                                c_in * input_h * input_w +
                                in_h * input_w + in_w] *
                           weight[c_out * in_channels * kernel_h * kernel_w +
                                  c_in * kernel_h * kernel_w +
                                  k_h * kernel_w + k_w];
                }
            }
        }
        output[batch * out_channels * output_h * output_w +
               c_out * output_h * output_w +
               out_y * output_w + out_x] = sum;
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                    torch::Tensor weight,
                                    int stride,
                                    int padding_h, int padding_w,
                                    int output_padding_h, int output_padding_w) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    auto input_h = input.size(2);
    auto input_w = input.size(3);
    auto out_channels = weight.size(0);
    auto output_h = (input_h - 1) * stride - 2 * padding_h + kernel_h + 2 * output_padding_h;
    auto output_w = (input_w - 1) * stride - 2 * padding_w + kernel_w + 2 * output_padding_w;

    auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());

    dim3 threads(32, 8, 8); // x, y, z (output channels)
    dim3 blocks(batch_size,
               (output_h + threads.y - 1) / threads.y,
               (output_w + threads.x - 1) / threads.x);

    // Launch kernel with 3D blocks (threadIdx.x: output_w, threadIdx.y: output_h, threadIdx.z: out_channels)
    ConvTranspose2dKernel<<<blocks, threads>>>(
        input.contiguous().data_ptr<scalar_t>(),
        weight.contiguous().data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size, in_channels, out_channels,
        kernel_h, kernel_w, input_h, input_w, output_h, output_w,
        stride, padding_h, padding_w, output_padding_h, output_padding_w);

    return output;
}
"""

conv_transpose2d_cpp = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int stride,
                                   int padding_h, int padding_w,
                                   int output_padding_h, int output_padding_w);
"""

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.bias = bias
        
        # Initialize weight
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        
        # Load the custom CUDA function
        self.conv_transpose2d_op = conv_transpose2d

    def forward(self, x):
        return self.conv_transpose2d_op.conv_transpose2d_cuda(
            x.cuda(),
            self.weight.cuda(),
            self.stride,
            self.padding[0], self.padding[1],
            self.output_padding[0], self.output_padding[1]
        )