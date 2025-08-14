import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

kernel_conv2d = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int THREADS = 256;
constexpr int SHM_PER_THREAD = 4;

template<typename scalar_t>
__global__ void optimized_conv2d_kernel(const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ weight,
                                       scalar_t* __restrict__ output,
                                       int batch_size, int in_channels, int out_channels,
                                       int input_height, int input_width,
                                       int kernel_h, int kernel_w,
                                       int stride, int pad_h, int pad_w,
                                       int dilation_h, int dilation_w,
                                       int groups) {

    const int output_h = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    const int output_w = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;
    const int channels_per_group = in_channels / groups;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int smem_offset = tx * SHM_PER_THREAD;

    __shared__ scalar_t shared_input[THREADS * SHM_PER_THREAD];

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int out_ch = threadIdx.x; out_ch < out_channels; out_ch += blockDim.x) {
            scalar_t sum = 0;
            for (int kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
                for (int kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
                    for (int in_group = 0; in_group < groups; ++in_group) {
                        int in_ch_base = in_group * channels_per_group;
                        for (int in_ch = 0; in_ch < channels_per_group; ++in_ch) {
                            int in_ch_idx = in_ch_base + in_ch;
                            int weight_offset = (out_ch * in_channels + in_ch_base + in_ch) * kernel_h * kernel_w;
                            weight_offset += kernel_y * kernel_w + kernel_x;

                            const scalar_t w = weight[weight_offset];

                            int input_y = (tx / output_w) * stride + kernel_y * dilation_h - pad_h;
                            int input_x = (tx % output_w) * stride + kernel_x * dilation_w - pad_w;
                            input_y = max(input_y, 0);
                            input_x = max(input_x, 0);

                            int input_offset = batch * in_channels * input_height * input_width +
                                              in_ch_idx * input_height * input_width +
                                              input_y * input_width + input_x;

                            sum += w * input[input_offset];
                        }
                    }
                }
            }
            output[batch * out_channels * output_h * output_w + out_ch * output_h * output_w + tx] = sum;
        }
    }
}

at::Tensor conv2d_forward(const at::Tensor input, const at::Tensor weight,
                         int stride, int padding, int dilation, int groups) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    auto output_height = (input_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    auto output_width = (input_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    auto output_size = {batch_size, out_channels, output_height, output_width};
    auto output = at::empty(output_size, input.options());

    dim3 blocks(1);
    dim3 threads(THREADS);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_forward", ([&]{
        optimized_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_h, kernel_w,
            stride, padding, padding,
            dilation, dilation,
            groups);
    }));

    return output;
}
"""

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
            
        # Compile CUDA kernels
        self.conv_op = load_inline(
            name='conv2d',
            cpp_sources='',
            cuda_sources=kernel_conv2d,
            functions=['conv2d_forward'],
            verbose=False
        )
        
    def forward(self, x):
        output = self.conv_op.conv2d_forward(
            x, self.weight, self.stride, self.padding,
            self.dilation, self.groups
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output