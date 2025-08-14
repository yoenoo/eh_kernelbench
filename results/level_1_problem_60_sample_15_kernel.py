import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Conv3d
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv3d_forward_kernel(const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> input,
                                     const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> weight,
                                     torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> output,
                                     int in_channels, int out_channels, int kernel_d, int kernel_h, int kernel_w,
                                     int stride, int pad_d, int pad_h, int pad_w,
                                     int dilation_d, int dilation_h, int dilation_w) {
    const int batch_size = input.size(0);
    const int input_d = input.size(2);
    const int input_h = input.size(3);
    const int input_w = input.size(4);
    const int output_d = output.size(2);
    const int output_h = output.size(3);
    const int output_w = output.size(4);

    CUDA_1D_KERNEL_LOOP(index, batch_size * out_channels * output_d * output_h * output_w) {
        int w = index % output_w;
        int h = (index / output_w) % output_h;
        int d = (index / (output_w * output_h)) % output_d;
        int c_out = (index / (output_w * output_h * output_d)) % out_channels;
        int n = index / (out_channels * output_d * output_h * output_w);

        scalar_t val = 0;
        for (int k_d = 0; k_d < kernel_d; ++k_d) {
            for (int k_h = 0; k_h < kernel_h; ++k_h) {
                for (int k_w = 0; k_w < kernel_w; ++k_w) {
                    // Compute input positions
                    int in_d = d * stride - pad_d + k_d * dilation_d;
                    int in_h = h * stride - pad_h + k_h * dilation_h;
                    int in_w = w * stride - pad_w + k_w * dilation_w;
                    
                    // Check boundaries
                    if (in_d >= 0 && in_d < input_d && in_h >=0 && in_h < input_h && in_w >=0 && in_w < input_w) {
                        for (int c_in = 0; c_in < in_channels; ++c_in) {
                            val += input[n][c_in][in_d][in_h][in_w] * weight[c_out][c_in][k_d][k_h][k_w];
                        }
                    }
                }
            }
        }
        output[n][c_out][d][h][w] = val;
    }
}

torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding_d, int padding_h, int padding_w,
                                int dilation_d, int dilation_h, int dilation_w) {
    // Output dimensions calculation
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    int input_d = input.size(2);
    int input_h = input.size(3);
    int input_w = input.size(4);

    int output_d = (input_d + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) / stride + 1;
    int output_h = (input_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int output_w = (input_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::zeros({input.size(0), out_channels, output_d, output_h, output_w}, input.options());

    const int threads = 256;
    int total_elements = output.numel();
    int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_forward_cuda", ([&] {
        conv3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            in_channels, out_channels, kernel_d, kernel_h, kernel_w,
            stride, padding_d, padding_h, padding_w,
            dilation_d, dilation_h, dilation_w
        );
    }));

    return output;
}
"""

conv3d_cpp_source = (
    "torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding_d, int padding_h, int padding_w, int dilation_d, int dilation_h, int dilation_w);"
)

# Compile the custom CUDA kernel
custom_conv3d = load_inline(
    name="custom_conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if isinstance(padding, int) else padding[0]
        self.dilation = dilation if isinstance(dilation, int) else dilation[0]
        self.groups = groups
        self.bias = bias

        # Initialize weights similar to PyTorch's Conv3d
        kernel_d, kernel_h, kernel_w = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_d, kernel_h, kernel_w))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias_weight = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_weight, -bound, bound)
        else:
            self.bias_weight = None

    def forward(self, x):
        # Extract parameters
        padding_d = self.padding if isinstance(self.padding, int) else self.padding[0]
        padding_h = self.padding if isinstance(self.padding, int) else self.padding[1]
        padding_w = self.padding if isinstance(self.padding, int) else self.padding[2]
        dilation_d = self.dilation if isinstance(self.dilation, int) else self.dilation[0]
        dilation_h = self.dilation if isinstance(self.dilation, int) else self.dilation[1]
        dilation_w = self.dilation if isinstance(self.dilation, int) else self.dilation[2]

        output = custom_conv3d.conv3d_forward_cuda(
            x,
            self.weight,
            self.stride,
            padding_d,
            padding_h,
            padding_w,
            dilation_d,
            dilation_h,
            dilation_w
        )

        if self.bias:
            output = output + self.bias_weight.view(1, -1, 1, 1, 1)

        return output