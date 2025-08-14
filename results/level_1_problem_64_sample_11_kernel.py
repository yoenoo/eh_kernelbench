import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weights similar to nn.ConvTranspose1d
        weight_shape = (in_channels, out_channels // groups, kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Load custom CUDA kernel
        self.conv_transpose_kernel = load_inline(
            name="conv_transpose_1d",
            cuda_sources=f"""
            #include <torch/extension.h>
            #include <cuda.h>
            #include <cuda_runtime.h>

            __global__ void conv_transpose_1d_kernel(
                const float* input,
                const float* weight,
                float* output,
                int batch_size,
                int in_channels,
                int out_channels,
                int kernel_size,
                int input_length,
                int output_length,
                int stride,
                int padding,
                int output_padding,
                int groups
            ) {{
                // Compute output indices
                int batch_idx = blockIdx.x;
                int out_channel = blockIdx.y * blockDim.x + threadIdx.x;
                int out_pos = blockIdx.z;

                if (out_channel >= out_channels || out_pos >= output_length) {{
                    return;
                }}

                // Compute the effective input position
                int effective_out_pos = out_pos + padding - output_padding;
                int input_start = (effective_out_pos) / stride;
                if (effective_out_pos % stride != 0) {{
                    return;  // This position is contributed by a non-integer shift, thus ignored in standard transposed conv
                }}
                input_start = effective_out_pos / stride - (kernel_size - 1);

                float val = 0.0;
                for (int k = 0; k < kernel_size; ++k) {{
                    int input_pos = input_start + k;
                    if (input_pos < 0 || input_pos >= input_length) {{
                        continue;
                    }}
                    for (int g = 0; g < groups; ++g) {{
                        int in_channel_group = (out_channel / (out_channels / groups)) * (in_channels / groups) + g;
                        val += weight[g * in_channels * out_channels + in_channel_group * kernel_size * out_channels + k * out_channels + out_channel] *
                               input[batch_idx * in_channels * input_length + in_channel_group * input_length + input_pos];
                    }}
                }}
                if (bias) {{
                    val += bias[out_channel];
                }}
                output[batch_idx * out_channels * output_length + out_channel * output_length + out_pos] = val;
            }}

            torch::Tensor conv_transpose_1d(
                torch::Tensor input,
                torch::Tensor weight,
                torch::Tensor bias,
                int stride,
                int padding,
                int output_padding,
                int groups
            ) {{
                const int batch_size = input.size(0);
                const int in_channels = input.size(1);
                const int input_length = input.size(2);
                const int out_channels = weight.size(1) * groups; // Since weight is [in_channels, out_channels_per_group, kernel_size]
                const int kernel_size = weight.size(2);

                // Compute output length
                const int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;

                auto output = torch::empty({{batch_size, out_channels, output_length}}, input.options());

                dim3 threads(256);
                dim3 blocks(batch_size, (out_channels + threads.x - 1) / threads.x, output_length);

                // Launch kernel
                conv_transpose_1d_kernel<<<blocks, threads>>>(
                    input.data_ptr<float>(),
                    weight.data_ptr<float>(),
                    output.data_ptr<float>(),
                    batch_size,
                    in_channels,
                    out_channels,
                    kernel_size,
                    input_length,
                    output_length,
                    stride,
                    padding,
                    output_padding,
                    groups
                );

                return output;
            }}
            """,
            functions=["conv_transpose_1d"],
            verbose=True
        )

    def reset_parameters(self):
        # Initialization similar to PyTorch's ConvTranspose1d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Reshape weight to match the kernel dimensions expected by the CUDA kernel
        weight = self.weight.permute(1, 0, 2).contiguous()  # [out_channels_per_group, in_channels_per_group, kernel_size]
        bias = self.bias.data if self.bias is not None else torch.empty(0)
        return self.conv_transpose_kernel.conv_transpose_1d(
            x.cuda(),
            weight.cuda(),
            bias.cuda() if self.bias is not None else torch.empty(0),
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )