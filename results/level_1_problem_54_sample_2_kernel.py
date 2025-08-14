import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized Conv3d
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void conv3d_forward_kernel(const scalar_t* __restrict__ input,
                                     const scalar_t* __restrict__ weight,
                                     scalar_t* __restrict__ output,
                                     const int batch_size,
                                     const int in_channels,
                                     const int depth,
                                     const int width,
                                     const int height,
                                     const int out_channels,
                                     const int kernel_size,
                                     const int stride,
                                     const int padding,
                                     const int dilation,
                                     const int groups,
                                     const int depth_out,
                                     const int width_out,
                                     const int height_out) {

    const int output_size = depth_out * width_out * height_out;
    const int group_channels = in_channels / groups;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batch_size * out_channels * output_size; idx += blockDim.x * gridDim.x) {
        int out_depth_idx = idx / (out_channels * width_out * height_out);
        int rest = idx % (out_channels * width_out * height_out);
        int out_channel = rest / (width_out * height_out);
        int out_pos = rest % (width_out * height_out);
        int out_width = out_pos / height_out;
        int out_height = out_pos % height_out;

        const int in_group = out_channel / (out_channels / groups);
        const int out_channel_in_group = out_channel % (out_channels / groups);

        scalar_t sum = 0;
        for (int d = 0; d < kernel_size; ++d) {
            for (int h = 0; h < kernel_size; ++h) {
                for (int w = 0; w < kernel_size; ++w) {
                    int input_d = out_depth_idx * stride + d * dilation - padding;
                    int input_w = out_width * stride + w * dilation - padding;
                    int input_h = out_height * stride + h * dilation - padding;
                    if (input_d < 0 || input_d >= depth || input_w < 0 || input_w >= width || input_h < 0 || input_h >= height) {
                        continue;
                    }
                    for (int c = 0; c < group_channels; ++c) {
                        sum += input[out_depth_idx * in_channels * depth * width * height + 
                                    (in_group * group_channels + c) * depth * width * height +
                                    c * depth * width * height + 
                                    input_d * width * height + 
                                    input_w * height + 
                                    input_h] *
                               weight[in_group * out_channels/group_channels * kernel_size*kernel_size*kernel_size * group_channels + 
                                      out_channel_in_group * kernel_size*kernel_size*kernel_size * group_channels +
                                      (d * kernel_size*kernel_size + h * kernel_size + w) * group_channels + 
                                      c];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor conv3d_forward(torch::Tensor input, torch::Tensor weight,
                            int batch_size, int in_channels, int depth, int width, int height,
                            int out_channels, int kernel_size, int stride, int padding, int dilation, int groups) {

    const int depth_out = (depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int width_out = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int height_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output_size = torch::IntArrayRef({batch_size, out_channels, depth_out, width_out, height_out});
    auto output = torch::empty(output_size, input.options());

    const int threads = 256;
    dim3 blocks = (batch_size * out_channels * depth_out * width_out * height_out + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_forward", ([&] {
        conv3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            depth,
            width,
            height,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            depth_out,
            width_out,
            height_out
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_forward(torch::Tensor input,
                            torch::Tensor weight,
                            int batch_size,
                            int in_channels,
                            int depth,
                            int width,
                            int height,
                            int out_channels,
                            int kernel_size,
                            int stride,
                            int padding,
                            int dilation,
                            int groups);
"""

conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, width, height = x.size()
        output = conv3d.conv3d_forward(
            x.cuda(),
            self.weight.cuda(),
            batch_size,
            self.in_channels,
            depth,
            width,
            height,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output