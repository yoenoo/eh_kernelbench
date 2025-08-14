import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias similar to ConvTranspose2d
        kernel_h, kernel_w = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_h, kernel_w))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Compile custom kernel
        self.custom_conv_t = load_inline(
            name="conv_transpose2d_custom",
            cuda_sources=f"""
            #include <torch/extension.h>
            #include <cuda.h>
            #include <cuda_runtime.h>

            template <typename scalar_t>
            __global__ void conv_transpose2d_kernel(
                const torch::PackedTensorAccessor<scalar_t,4> input,
                const torch::PackedTensorAccessor<scalar_t,4> weight,
                torch::PackedTensorAccessor<scalar_t,4> output,
                int kernel_h, int kernel_w, int stride, int padding, int output_padding,
                int batch_size, int in_channels, int out_channels,
                int input_h, int input_w, int output_h, int output_w,
                int groups)
            {{
                const int B = blockIdx.z;
                const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
                const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (out_y >= output_h || out_x >= output_w) return;

                for (int out_ch = threadIdx.z; out_ch < out_channels; out_ch += blockDim.z) {{
                    scalar_t sum = 0;
                    for (int g = 0; g < groups; ++g) {{
                        int in_ch_start = g * in_channels / groups;
                        for (int k_y = 0; k_y < kernel_h; ++k_y) {{
                            for (int k_x = 0; k_x < kernel_w; ++k_x) {{
                                int in_y = (out_y + padding - k_y) / stride;
                                int in_x = (out_x + padding - k_x) / stride;
                                if ((out_y + padding - k_y) % stride == 0 && (out_x + padding - k_x) % stride == 0 &&
                                    in_y >= 0 && in_y < input_h &&
                                    in_x >= 0 && in_x < input_w) {{
                                    for (int in_ch = in_ch_start; in_ch < in_ch_start + in_channels/groups; ++in_ch) {{
                                        sum += input[B][in_ch][in_y][in_x] * 
                                               weight[out_ch][in_ch][k_y][k_x];
                                    }}
                                }}
                            }}
                        }}
                    }}
                    if (this->bias != nullptr) sum += bias[out_ch];
                    output[B][out_ch][out_y][out_x] = sum;
                }}
            }}

            torch::Tensor custom_conv_transpose2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                                int kernel_h, int kernel_w, int stride, int padding, int output_padding,
                                                int groups) {{
                const int batch_size = input.size(0);
                const int in_channels = input.size(1);
                const int input_h = input.size(2);
                const int input_w = input.size(3);

                const int output_h = (input_h - 1) * stride - 2 * padding + kernel_h + output_padding;
                const int output_w = (input_w - 1) * stride - 2 * padding + kernel_w + output_padding;
                
                auto output = torch::empty({{batch_size, out_channels, output_h, output_w}}, 
                                          device=input.device(), dtype=input.dtype());

                dim3 threads(8, 8, 8);
                dim3 blocks((output_w + threads.x -1)/threads.x, 
                            (output_h + threads.y -1)/threads.y, 
                            batch_size);

                AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d", ([&] {
                    conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
                        input.packed_accessor<scalar_t,4>(),
                        weight.packed_accessor<scalar_t,4>(),
                        output.packed_accessor<scalar_t,4>(),
                        kernel_h, kernel_w, stride, padding, output_padding,
                        batch_size, in_channels, out_channels,
                        input_h, input_w, output_h, output_w,
                        groups);
                }));

                return output;
            }}
            """,
            functions=["custom_conv_transpose2d"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_conv_t.custom_conv_transpose2d(
            x, 
            self.weight, 
            self.bias if self.bias is not None else torch.empty(0),
            self.kernel_size[0], 
            self.kernel_size[1],
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide parameters for initialization