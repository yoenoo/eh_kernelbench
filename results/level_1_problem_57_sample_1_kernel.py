import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weights and bias similar to ConvTranspose2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Load custom CUDA kernel
        conv_transpose_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void conv_transpose2d_kernel(const scalar_t* input,
                                                const scalar_t* weight,
                                                scalar_t* output,
                                                int batch_size,
                                                int in_channels,
                                                int out_channels,
                                                int kernel_size,
                                                int input_height,
                                                int input_width,
                                                int output_height,
                                                int output_width,
                                                int stride,
                                                int padding,
                                                int output_padding,
                                                int groups) {

            const int H_out = output_height;
            const int W_out = output_width;
            const int H_in = input_height;
            const int W_in = input_width;

            const int group_size = in_channels / groups;
            const int group_out_size = out_channels / groups;

            int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (output_idx >= batch_size * out_channels * H_out * W_out) {
                return;
            }

            int w_out = output_idx % W_out;
            int h_out = (output_idx / W_out) % H_out;
            int c_out = (output_idx / (W_out * H_out)) % out_channels;
            int n = output_idx / (out_channels * H_out * W_out);

            int group_id = c_out / group_out_size;
            int c_in_base = group_id * group_size;

            scalar_t val = 0;
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int h_in = (h_out + padding - kh) / stride;
                    int w_in = (w_out + padding - kw) / stride;
                    if ((h_out + padding - kh) % stride != 0 ||
                        (w_out + padding - kw) % stride != 0) {
                        continue;
                    }
                    if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) {
                        continue;
                    }
                    for (int c_in = c_in_base; c_in < c_in_base + group_size; ++c_in) {
                        val += input[n * in_channels * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in] *
                               weight[c_out * kernel_size * kernel_size * group_size + (kh * kernel_size + kw) * group_size + (c_in - c_in_base)];
                    }
                }
            }

            if (bias) {
                val += bias[c_out];
            }
            output[output_idx] = val;
        }

        torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                           torch::Tensor weight,
                                           torch::optional<torch::Tensor> bias,
                                           int stride,
                                           int padding,
                                           int output_padding,
                                           int groups) {

            auto batch_size = input.size(0);
            auto in_channels = input.size(1);
            auto out_channels = weight.size(0);
            auto kernel_size = weight.size(2);
            auto input_height = input.size(2);
            auto input_width = input.size(3);
            auto output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
            auto output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

            auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

            int threads = 256;
            int elements = batch_size * out_channels * output_height * output_width;
            int blocks = (elements + threads - 1) / threads;

            AT_DISPATCH_ALL_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
                conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
                    input.data<scalar_t>(),
                    weight.data<scalar_t>(),
                    output.data<scalar_t>(),
                    batch_size,
                    in_channels,
                    out_channels,
                    kernel_size,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    stride,
                    padding,
                    output_padding,
                    groups);
            }));

            cudaDeviceSynchronize();
            return output;
        }
        """

        conv_transpose_cpp_source = """
        torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                           torch::Tensor weight,
                                           torch::optional<torch::Tensor> bias,
                                           int stride,
                                           int padding,
                                           int output_padding,
                                           int groups);
        """

        self.conv_transpose_op = load_inline(
            name="conv_transpose",
            cpp_sources=conv_transpose_cpp_source,
            cuda_sources=conv_transpose_source,
            functions=["conv_transpose2d_cuda"],
            verbose=True
        )

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.bias is not None else torch.tensor([])
        return self.conv_transpose_op.conv_transpose2d_cuda(
            x,
            self.weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )