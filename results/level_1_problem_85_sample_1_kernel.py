import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ weights,
                                       scalar_t* __restrict__ output,
                                       int batch_size, int in_channels,
                                       int input_height, int input_width,
                                       int kernel_h, int kernel_w,
                                       int stride_h, int stride_w,
                                       int padding_h, int padding_w,
                                       int dilation_h, int dilation_w,
                                       int output_height, int output_width) {
    int n = blockIdx.x;
    int c = blockIdx.y;

    const scalar_t* input_row = input + n * in_channels * input_height * input_width + c * input_height * input_width;
    const scalar_t* weight_row = weights + c * kernel_h * kernel_w;

    int output_index = n * in_channels * output_height * output_width + c * output_height * output_width;

    for (int ph = threadIdx.y; ph < kernel_h; ph += blockDim.y) {
        for (int pw = threadIdx.x; pw < kernel_w; pw += blockDim.x) {
            int kh = ph * dilation_h;
            int kw = pw * dilation_w;

            scalar_t weight_val = weight_row[ph * kernel_w + pw];

            for (int oh = 0; oh < output_height; oh += stride_h) {
                for (int ow = 0; ow < output_width; ow += stride_w) {
                    int ih = -padding_h + oh + kh;
                    int iw = -padding_w + ow + kw;

                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        int input_offset = ih * input_width + iw;
                        int output_offset = (oh / stride_h) * output_width + (ow / stride_w);
                        atomicAdd(output + output_index + output_offset, input_row[input_offset] * weight_val);
                    }
                }
            }
        }
    }
}

torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weights,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int kernel_h = weights.size(2);
    const int kernel_w = weights.size(3);

    // Compute output dimensions
    int output_height = (input_height + 2 * padding_h -
                        dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int output_width = (input_width + 2 * padding_w -
                       dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch_size, in_channels, output_height, output_width},
                              input.options());

    dim3 threads(32, 8);
    dim3 blocks(batch_size, in_channels);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "depthwise_conv2d_forward", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weights.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, input_height, input_width,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w,
            output_height, output_width);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

depthwise_conv_cpp_source = """
torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weights,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w);
"""

depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int,
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0,
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        
        # Initialize weights similar to PyTorch's Conv2d
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = depthwise_conv.depthwise_conv2d_forward(
            x, self.weight, self.stride_h, self.stride_w,
            self.padding_h, self.padding_w, self.dilation_h, self.dilation_w)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out