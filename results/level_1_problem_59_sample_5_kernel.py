import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv3d_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv3d_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int kernel_height,
    const int kernel_width,
    const int stride,
    const int padding,
    const int dilation) {

    const int output_depth = input_depth;
    const int output_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    CUDA_1D_KERNEL_LOOP(output_idx, batch_size * out_channels * output_depth * output_height * output_width) {
        int w = output_idx % output_width;
        int h = (output_idx / output_width) % output_height;
        int d = (output_idx / (output_width * output_height)) % output_depth;
        int c_out = (output_idx / (output_depth * output_height * output_width)) % out_channels;
        int n = output_idx / (out_channels * output_depth * output_height * output_width);

        scalar_t val = 0;
        for (int k = 0; k < kernel_height; ++k) {
            for (int l = 0; l < kernel_width; ++l) {
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    const int input_h = h * stride - padding + dilation * k;
                    const int input_w = w * stride - padding + dilation * l;
                    if (input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                        const int input_offset = n * in_channels * input_depth * input_height * input_width +
                                                 c_in * input_depth * input_height * input_width +
                                                 d * input_height * input_width +
                                                 input_h * input_width + input_w;
                        const int weight_offset = c_out * in_channels * kernel_height * kernel_width +
                                                  c_in * kernel_height * kernel_width +
                                                  k * kernel_width + l;
                        val += input[input_offset] * weight[weight_offset];
                    }
                }
            }
        }
        output[output_idx] = val;
    }
}

torch::Tensor conv3d_forward(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_depth = input.size(4);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_height = weight.size(2);
    const auto kernel_width = weight.size(3);

    auto output_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    auto output_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;
    auto output_depth = input_depth;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width, output_depth}, input.options());

    dim3 blocks(TORCH_CAFFE_GET_BLOCKS(output.numel()));
    dim3 threads(TORCHThreadPerBlock);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_forward", ([&] {
        conv3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_depth,
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            stride,
            padding,
            dilation
        );
    }));

    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_forward(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);
"""

conv3d_ops = load_inline(
    name='conv3d_ops',
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_kernel_source,
    functions=['conv3d_forward'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size, 1)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights like PyTorch Conv3d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
            
        self.conv3d_forward = conv3d_ops.conv3d_forward

    def forward(self, x):
        output = self.conv3d_forward(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        return output