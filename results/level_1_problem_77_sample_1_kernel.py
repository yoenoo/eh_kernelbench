import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose_3d_kernel(const scalar_t* input,
                                        const scalar_t* weight,
                                        scalar_t* output,
                                        int batch_size,
                                        int in_channels,
                                        int out_channels,
                                        int input_depth,
                                        int input_height,
                                        int input_width,
                                        int kernel_size,
                                        int output_depth,
                                        int output_height,
                                        int output_width,
                                        int stride,
                                        int padding,
                                        int dilation) {
    const int output_size = output_depth * output_height * output_width;
    const int batch_idx = blockIdx.x;
    const int out_depth = threadIdx.z;
    const int out_row = threadIdx.y;
    const int out_col = threadIdx.x;
    const int out_pos = out_depth * output_height * output_width + out_row * output_width + out_col;

    if (out_pos >= output_size) return;

    const int in_depth = (out_depth + padding) / stride - padding;
    const int in_row = (out_row + padding) / stride - padding;
    const int in_col = (out_col + padding) / stride - padding;

    scalar_t val = 0;
    for (int k_d = 0; k_d < kernel_size; ++k_d) {
        for (int k_h = 0; k_h < kernel_size; ++k_h) {
            for (int k_w = 0; k_w < kernel_size; ++k_w) {
                int input_d = in_depth + dilation * k_d;
                int input_r = in_row + dilation * k_h;
                int input_c = in_col + dilation * k_w;

                if (input_d < 0 || input_d >= input_depth ||
                    input_r < 0 || input_r >= input_height ||
                    input_c < 0 || input_c >= input_width) {
                    continue;
                }

                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int oc = 0; oc < out_channels; ++oc) {
                        int weight_idx = oc * in_channels * kernel_size*kernel_size*kernel_size +
                                        ic * kernel_size*kernel_size*kernel_size +
                                        k_d * kernel_size*kernel_size +
                                        k_h * kernel_size +
                                        k_w;
                        int input_idx = batch_idx * in_channels * input_depth * input_height * input_width +
                                        ic * input_depth * input_height * input_width +
                                        input_d * input_height * input_width +
                                        input_r * input_width +
                                        input_c;
                        int output_idx = batch_idx * out_channels * output_depth * output_height * output_width +
                                        oc * output_depth * output_height * output_width +
                                        out_depth * output_height * output_width +
                                        out_row * output_width +
                                        out_col;

                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    output[output_idx] = val;
}

torch::Tensor conv_transpose_3d_cuda(torch::Tensor input,
                                    torch::Tensor weight,
                                    int kernel_size,
                                    int stride,
                                    int padding,
                                    int dilation) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0);
    const auto input_depth = input.size(2);
    const auto input_height = input.size(3);
    const auto input_width = input.size(4);

    const auto output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + 2 * padding;
    const auto output_height = (input_height - 1) * stride - 2 * padding + kernel_size + 2 * padding;
    const auto output_width = (input_width - 1) * stride - 2 * padding + kernel_size + 2 * padding;

    auto output_options = torch::TensorOptions().like(input);
    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, output_options);

    dim3 threads(32, 32, 8);
    dim3 blocks(batch_size, 1, 1);

    const int total_threads = threads.x * threads.y * threads.z;
    AT_ASSERT(total_threads <= 1024, "Thread count exceeds maximum");

    const int kernel_size_3d = kernel_size * kernel_size * kernel_size;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose_3d_cuda", ([&]{
        conv_transpose_3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_depth,
            input_height,
            input_width,
            kernel_size,
            output_depth,
            output_height,
            output_width,
            stride,
            padding,
            dilation);
    }));

    return output;
}
"""

conv_transpose_3d_cpp_source = """
torch::Tensor conv_transpose_3d_cuda(torch::Tensor input,
                                    torch::Tensor weight,
                                    int kernel_size,
                                    int stride,
                                    int padding,
                                    int dilation);
"""

conv_transpose_3d = load_inline(
    name="conv_transpose_3d",
    cpp_sources=conv_transpose_3d_cpp_source,
    cuda_sources=conv_transpose_3d_source,
    functions=["conv_transpose_3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        out = conv_transpose_3d.conv_transpose_3d_cuda(
            x,
            self.weight,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1, 1)
        return out