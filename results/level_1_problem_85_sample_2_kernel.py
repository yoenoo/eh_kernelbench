import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::Default_GPU> input,
                                       const torch::PackedTensorAccessor<scalar_t,4,torch::Default_GPU> weight,
                                       torch::PackedTensorAccessor<scalar_t,4,torch::Default_GPU> output,
                                       int kernel_h, int kernel_w,
                                       int stride_h, int stride_w,
                                       int padding_h, int padding_w,
                                       int dilation_h, int dilation_w) {
    const int n = blockIdx.z;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (h_out >= output.size(2) || w_out >= output.size(3)) {
        return;
    }

    scalar_t sum = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int h_in = -padding_h + h_out * stride_h + kh * dilation_h;
            const int w_in = -padding_w + w_out * stride_w + kw * dilation_w;
            if (h_in >= 0 && h_in < input.size(2) && w_in >= 0 && w_in < input.size(3)) {
                sum += input[n][0][h_in][w_in] * weight[0][kh][kw];
            }
        }
    }
    output[n][0][h_out][w_out] = sum;
}

std::tuple<torch::Tensor> depthwise_convolution_cuda(torch::Tensor input, torch::Tensor weight,
                                                    int kernel_h, int kernel_w,
                                                    int stride_h, int stride_w,
                                                    int padding_h, int padding_w,
                                                    int dilation_h, int dilation_w) {

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    const int output_h = (input_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int output_w = (input_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch, channels, output_h, output_w}, input.options());

    dim3 threads(16, 16);
    dim3 blocks(
        (output_w + threads.x - 1) / threads.x,
        (output_h + threads.y - 1) / threads.y,
        batch * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_convolution_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::Default_GPU>(),
            weight.packed_accessor<scalar_t,4,torch::Default_GPU>(),
            output.packed_accessor<scalar_t,4,torch::Default_GPU>(),
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w
        );
    }));

    return output;
}
"""

depthwise_conv_cpp_source = """
std::tuple<torch::Tensor> depthwise_convolution_cuda(torch::Tensor input, torch::Tensor weight,
                                                    int kernel_h, int kernel_w,
                                                    int stride_h, int stride_w,
                                                    int padding_h, int padding_w,
                                                    int dilation_h, int dilation_w);
"""

depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_convolution_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int,
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0,
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size_h, kernel_size_w)
        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)
        self.dilation = (dilation_h, dilation_w)
        self.groups = groups
        self.bias = bias

        # Initialize weights similar to PyTorch's Conv2d
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size_h, kernel_size_w))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return depthwise_conv.depthwise_convolution_cuda(
            x,
            self.weight,
            self.kernel_size[0], self.kernel_size[1],
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )[0]