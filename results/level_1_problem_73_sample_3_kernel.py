import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 3D Transposed Convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void ConvTranspose3DKernel(const scalar_t* __restrict__ input,
                                     const scalar_t* __restrict__ weight,
                                     scalar_t* __restrict__ output,
                                     const int batch_size,
                                     const int in_channels,
                                     const int out_channels,
                                     const int depth_in,
                                     const int height_in,
                                     const int width_in,
                                     const int kernel_size,
                                     const int stride,
                                     const int padding,
                                     const int depth_out,
                                     const int height_out,
                                     const int width_out,
                                     const int groups) {
    // Calculate output element indices
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int d = blockIdx.z * blockDim.z + threadIdx.z;
    const int batch = blockIdx.w;

    if (w >= width_out || h >= height_out || d >= depth_out || batch >= batch_size) {
        return;
    }

    scalar_t val = 0;
    const int in_group = in_channels / groups;
    const int out_group = out_channels / groups;

    // Iterate over the groups
    for (int g = 0; g < groups; ++g) {
        // Compute the input position based on stride and padding
        int d_in = (d - padding) / stride;
        int h_in = (h - padding) / stride;
        int w_in = (w - padding) / stride;

        if (d_in < 0 || d_in >= depth_in || h_in < 0 || h_in >= height_in || w_in < 0 || w_in >= width_in) {
            continue;
        }

        // Iterate over kernel elements
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    // Compute input index with current kernel element
                    int d_kernel = kd + d_in * stride;
                    int h_kernel = kh + h_in * stride;
                    int w_kernel = kw + w_in * stride;

                    if (d_kernel >= depth_out || h_kernel >= height_out || w_kernel >= width_out) {
                        continue;
                    }

                    // Input channel index within group
                    for (int ic = 0; ic < in_group; ++ic) {
                        int input_channel = g * in_group + ic;
                        // Output channel loops over each out channel in the group
                        for (int oc = 0; oc < out_group; ++oc) {
                            int output_channel = g * out_group + oc;

                            // Compute weight index: (oc, ic, kd, kh, kw)
                            int weight_offset = oc * in_group * kernel_size * kernel_size * kernel_size +
                                                ic * kernel_size * kernel_size * kernel_size +
                                                kd * kernel_size * kernel_size +
                                                kh * kernel_size +
                                                kw;

                            // Input index: (batch, input_channel, d_in, h_in, w_in)
                            int input_offset = batch * in_channels * depth_in * height_in * width_in +
                                              input_channel * depth_in * height_in * width_in +
                                              d_in * height_in * width_in +
                                              h_in * width_in +
                                              w_in;

                            // Output index: (batch, output_channel, d, h, w)
                            int output_offset = batch * out_channels * depth_out * height_out * width_out +
                                               output_channel * depth_out * height_out * width_out +
                                               d * height_out * width_out +
                                               h * width_out +
                                               w;

                            val += weight[weight_offset] * input[input_offset];
                        }
                    }
                }
            }
        }
    }
    output[output_offset] = val;
}

std::tuple<torch::Tensor> conv_transpose3d_cuda(torch::Tensor input,
                                               torch::Tensor weight,
                                               int stride,
                                               int padding,
                                               int groups) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto depth_in = input.size(2);
    const auto height_in = input.size(3);
    const auto width_in = input.size(4);

    const auto out_channels = weight.size(0) * groups;
    const auto kernel_size = weight.size(2); // Assume square kernel
    const auto depth_out = (depth_in - 1) * stride - 2 * padding + kernel_size + 1;
    const auto height_out = (height_in - 1) * stride - 2 * padding + kernel_size + 1;
    const auto width_out = (width_in - 1) * stride - 2 * padding + kernel_size + 1;

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    dim3 threads(16, 16, 2);
    dim3 blocks(
        (width_out + threads.x - 1) / threads.x,
        (height_out + threads.y - 1) / threads.y,
        (depth_out + threads.z - 1) / threads.z,
        batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_cuda", ([&] {
        ConvTranspose3DKernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, out_channels,
            depth_in, height_in, width_in,
            kernel_size, stride, padding,
            depth_out, height_out, width_out,
            groups);
    }));

    return output;
}
"""

cpp_source = """
std::tuple<torch::Tensor> conv_transpose3d_cuda(torch::Tensor input,
                                               torch::Tensor weight,
                                               int stride,
                                               int padding,
                                               int groups);
"""

# Load the CUDA extension
conv_transposed3d = load_inline(
    name="conv_transposed3d",
    cpp_sources=cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.groups = groups
        self.stride = stride
        self.padding = padding
        # Initialize weight parameters as in PyTorch's ConvTranspose3d
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        output = conv_transposed3d.conv_transpose3d_cuda(
            x, self.weight, self.stride, self.padding, self.groups)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        return output