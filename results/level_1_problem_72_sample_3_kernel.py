import torch
import torch.nn as nn

from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose3D
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight, const scalar_t* __restrict__ bias,
    const int batch_size, const int in_channels, const int out_channels, 
    const int kernel_size_d, const int kernel_size_h, const int kernel_size_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int padding_d, const int padding_h, const int padding_w,
    const int output_padding_d, const int output_padding_h, const int output_padding_w,
    const int groups,
    const int input_depth, const int input_height, const int input_width,
    const int output_depth, const int output_height, const int output_width) {

    const int out_depth = output_depth;
    const int out_height = output_height;
    const int out_width = output_width;

    const int d_out = blockIdx.z;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = blockIdx.y / out_height;

    if (h_out >= out_height || w_out >= out_width || batch_id >= batch_size) {
        return;
    }

    scalar_t val = bias[batch_id * out_channels + threadIdx.z];

    const int in_group_channels = in_channels / groups;
    const int out_group_channels = out_channels / groups;

    for (int g = 0; g < groups; ++g) {
        int in_channel_offset = g * in_group_channels;
        int out_channel_offset = g * out_group_channels;
        for (int kd = 0; kd < kernel_size_d; ++kd) {
            const int d_in = (d_out + padding_d - kd) / stride_d;
            if (d_in < 0 || d_in >= input_depth) continue;

            for (int kh = 0; kh < kernel_size_h; ++kh) {
                const int h_in = (h_out + padding_h - kh) / stride_h;
                if (h_in < 0 || h_in >= input_height) continue;

                for (int kw = 0; kw < kernel_size_w; ++kw) {
                    const int w_in = (w_out + padding_w - kw) / stride_w;
                    if (w_in < 0 || w_in >= input_width) continue;

                    for (int in_c = 0; in_c < in_group_channels; ++in_c) {
                        int input_idx = batch_id * in_channels * input_depth * input_height * input_width
                            + (in_channel_offset + in_c) * input_depth * input_height * input_width
                            + d_in * input_height * input_width
                            + h_in * input_width
                            + w_in;

                        int weight_idx = out_channel_offset * kernel_size_d * kernel_size_h * kernel_size_w * in_group_channels
                            + in_c * kernel_size_d * kernel_size_h * kernel_size_w
                            + kd * kernel_size_h * kernel_size_w
                            + kh * kernel_size_w
                            + kw;

                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    int output_idx = batch_id * out_channels * output_depth * output_height * output_width
        + threadIdx.z * output_depth * output_height * output_width
        + d_out * output_height * output_width
        + h_out * output_width
        + w_out;

    out[output_idx] = val;
}

std::vector<int64_t> calculate_output_shape(int input_depth, int input_height, int input_width,
    int kernel_size_d, int kernel_size_h, int kernel_size_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w) {
    int output_depth = (input_depth - 1) * stride_d - 2 * padding_d + kernel_size_d + output_padding_d;
    int output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_size_h + output_padding_h;
    int output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_size_w + output_padding_w;
    return {output_depth, output_height, output_width};
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
        int stride_d, int stride_h, int stride_w,
        int padding_d, int padding_h, int padding_w,
        int output_padding_d, int output_padding_h, int output_padding_w,
        int groups) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_size_d = weight.size(2);
    const int kernel_size_h = weight.size(3);
    const int kernel_size_w = weight.size(4);

    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);

    auto output_dims = calculate_output_shape(
        input_depth, input_height, input_width,
        kernel_size_d, kernel_size_h, kernel_size_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w);

    const int output_depth = output_dims[0];
    const int output_height = output_dims[1];
    const int output_width = output_dims[2];

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, output_options);

    const dim3 threads(32, 8, 1);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        output_depth * batch_size);

    const int num_channels_per_block = out_channels / groups;

    const int total_threads_y = threads.y * ((output_height + threads.y - 1) / threads.y);
    const int total_threads_x = threads.x * ((output_width + threads.x - 1) / threads.x);

    // Bias handling
    auto bias_broadcast = bias.view({1, -1}).expand({batch_size, out_channels});

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose3d_cuda", ([&] {
        conv_transpose3d_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias_broadcast.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            kernel_size_d, kernel_size_h, kernel_size_w,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w,
            groups,
            input_depth, input_height, input_width,
            output_depth, output_height, output_width);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups);
"""

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weights and bias similar to nn.ConvTranspose3d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        output_padding_d, output_padding_h, output_padding_w = self.output_padding

        return conv_transpose3d.conv_transpose3d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.zeros(self.out_channels, device=x.device),
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w,
            self.groups
        )