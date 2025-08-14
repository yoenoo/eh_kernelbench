import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize parameters similar to ConvTranspose1d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # weights and bias similar to PyTorch's initialization
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Custom CUDA kernel definition for ConvTranspose1D
        self._initialize_parameters()
        self.custom_conv_transpose = load_inline(
            name="custom_conv_transpose",
            cuda_sources=f"""
                #include <torch/extension.h>
                #include <ATen/cuda/CUDAContext.h>
                #include <cuda_runtime.h>

                __global__ void conv_transpose1d_kernel(
                    const float* input, const float* weight, float* output,
                    int batch_size, int in_channels, int out_channels,
                    int input_length, int kernel_size, int stride,
                    int padding, int output_padding, int groups,
                    int output_length, bool has_bias, const float* bias) {{
                    
                    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (output_idx >= batch_size * out_channels * output_length)
                        return;

                    int b = output_idx / (out_channels * output_length);
                    int c_out = (output_idx / output_length) % out_channels;
                    int t_out = output_idx % output_length;

                    float acc = 0.0;
                    // Determine the group this output channel belongs to
                    int group_id = c_out / (out_channels / groups);
                    int group_in_channels = in_channels / groups;
                    int group_out_channels = out_channels / groups;
                    int c_in_start = group_id * group_in_channels;
                    int c_out_start = group_id * group_out_channels;

                    for (int k = 0; k < kernel_size; ++k) {{
                        int t_in = t_out * stride - padding + k - (kernel_size - 1);
                        if (t_in < 0 || t_in >= input_length) continue;
                        for (int c_in = c_in_start; c_in < c_in_start + group_in_channels; ++c_in) {{
                            int weight_idx = (c_out - c_out_start) * group_in_channels * kernel_size 
                                            + (c_in - c_in_start) * kernel_size + k;
                            acc += input[b * in_channels * input_length + c_in * input_length + t_in] 
                                * weight[weight_idx];
                        }}
                    }}
                    if (has_bias) {{
                        acc += bias[c_out];
                    }}
                    output[output_idx] = acc;
                }}

                at::Tensor conv_transpose1d_cuda(
                    at::Tensor input, at::Tensor weight, at::Tensor bias,
                    int stride, int padding, int output_padding,
                    int groups, int kernel_size, int output_length) {{
                    
                    const auto batch_size = input.size(0);
                    const auto in_channels = input.size(1);
                    const auto input_length = input.size(2);

                    auto output = at::empty({{batch_size, out_channels, output_length}}, input.type());
                    
                    dim3 threads(256);
                    dim3 blocks((batch_size * out_channels * output_length + threads.x - 1) / threads.x);
                    
                    AT_CUDA_CHECK(cudaGetLastError());
                    conv_transpose1d_kernel<<<blocks, threads, 0, at::cuda::current_stream()>>>(
                        input.data<float>(), weight.data<float>(), output.data<float>(),
                        batch_size, in_channels, out_channels,
                        input_length, kernel_size, stride,
                        padding, output_padding, groups,
                        output_length, bias.defined(), bias.data<float>());

                    return output;
                }}
            """,
            functions=["conv_transpose1d_cuda"],
            verbose=True
        )

    def _initialize_parameters(self):
        # Initialize weights using Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Compute output length as per PyTorch formula
        input_length = x.size(2)
        output_length = (input_length - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        # Ensure correct output length computation (matches PyTorch's output shape)
        
        # Call the custom CUDA kernel
        if self.bias is not None:
            output = self.custom_conv_transpose.conv_transpose1d_cuda(
                x, self.weight, self.bias, self.stride,
                self.padding, self.output_padding, self.groups,
                self.kernel_size, output_length
            )
        else:
            output = self.custom_conv_transpose.conv_transpose1d_cuda(
                x, self.weight, at::Tensor(), self.stride,
                self.padding, self.output_padding, self.groups,
                self.kernel_size, output_length
            )
        return output

    # Convenience function to get initialization parameters (for original test code compatibility)
    @classmethod
    def from_original_init(cls, in_channels, out_channels, kernel_size, **kwargs):
        return cls(in_channels, out_channels, kernel_size, **kwargs)