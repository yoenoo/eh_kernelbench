import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D convolution
conv2d_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

template <typename scalar_t>
__global__ void conv2d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,4> input,
                                     const torch::PackedTensorAccessor<scalar_t,4> weight,
                                     torch::PackedTensorAccessor<scalar_t,4> output,
                                     int kernel_h, int kernel_w,
                                     int stride, int padding_h, int padding_w,
                                     int dilation_h, int dilation_w) {

    const int B = blockIdx.z;
    const int C_out = blockIdx.y;
    const int Y = blockIdx.x * blockDim.y + threadIdx.y;
    const int X = threadIdx.x;

    if (Y >= output.size(2) || X >= output.size(3)) {
        return;
    }

    scalar_t sum = 0;
    for (int i = 0; i < weight.size(1); ++i) { // in_channels
        for (int ky = 0; ky < kernel_h; ++ky) {
            for (int kx = 0; kx < kernel_w; ++kx) {
                int h_in = -padding_h + Y * stride + dilation_h * ky;
                int w_in = -padding_w + X * stride + dilation_w * kx;
                if (h_in >= 0 && h_in < input.size(2) && w_in >=0 && w_in < input.size(3)) {
                    sum += weight[C_out][i][ky][kx] * input[B][i][h_in][w_in];
                }
            }
        }
    }
    output[B][C_out][Y][X] = sum;
}

std::vector<int64_t> compute_output_size(const torch::Tensor& input,
                                        const torch::Tensor& weight,
                                        int kernel_h, int kernel_w,
                                        int stride, int padding_h, int padding_w,
                                        int dilation_h, int dilation_w) {

    int batch_size = input.size(0);
    int out_channels = weight.size(0);
    int in_height = input.size(2);
    int in_width = input.size(3);

    int kernel_effective_h = dilation_h * (kernel_h - 1) + 1;
    int kernel_effective_w = dilation_w * (kernel_w - 1) + 1;

    int out_height = (in_height + 2 * padding_h - kernel_effective_h) / stride + 1;
    int out_width = (in_width + 2 * padding_w - kernel_effective_w) / stride + 1;

    return {batch_size, out_channels, out_height, out_width};
}

torch::Tensor custom_conv2d_forward(torch::Tensor input, torch::Tensor weight,
                                   int kernel_h, int kernel_w,
                                   int stride, int padding_h, int padding_w,
                                   int dilation_h, int dilation_w) {

    auto output_size = compute_output_size(input, weight, kernel_h, kernel_w,
                                          stride, padding_h, padding_w,
                                          dilation_h, dilation_w);

    auto output = torch::zeros(output_size, torch::device("cuda"));

    int blocksY = output_size[2];
    int blocksX = 1;
    dim3 threads(32, 8); // 256 threads per block
    dim3 blocks(output_size[3], blocksY, output_size[0] * output_size[1]);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv2d_forward", ([&] {
        conv2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            kernel_h, kernel_w,
            stride, padding_h, padding_w,
            dilation_h, dilation_w);
    }));

    return output;
}
"""

conv2d_cpp_source = """
#include <torch/extension.h>
torch::Tensor custom_conv2d_forward(torch::Tensor input, torch::Tensor weight,
                                   int kernel_h, int kernel_w,
                                   int stride, int padding_h, int padding_w,
                                   int dilation_h, int dilation_w);
"""

# Compile the inline CUDA code for 2D convolution
custom_conv2d = load_inline(
    name="custom_conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_kernel_source,
    functions=["custom_conv2d_forward"],
    verbose=True,
    extra_cflags=["-DUSE_CUDA"],
    extra_cuda_cflags=["-lineinfo"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):

        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.bias = bias

        # Initialize convolution weights
        kernel_h, kernel_w = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_h, kernel_w))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Load custom CUDA functions
        self.custom_conv2d = custom_conv2d

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.custom_conv2d.custom_conv2d_forward(
            x.cuda(),
            self.weight.cuda(),
            self.kernel_size[0], self.kernel_size[1],
            self.stride, self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output