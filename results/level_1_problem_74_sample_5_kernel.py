import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize weights and bias similar to ConvTranspose1d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

        # Compile the custom CUDA kernel
        self.conv_transpose_cuda = load_inline(
            name="conv_transpose_cuda",
            cpp_sources="""
            torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                             int in_channels, int out_channels, int kernel_size, int stride,
                                             int padding, int dilation, bool has_bias);
            """,
            cuda_sources=f"""
            #include <torch/extension.h>
            #include <ATen/ATen.h>
            #include <ATen/cuda/CUDAContext.h>
            #include <cuda_runtime.h>

            template <typename scalar_t>
            __global__ void conv_transpose_1d_kernel(
                const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> input,
                const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> weight,
                torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> output,
                const int in_channels, const int out_channels, const int kernel_size,
                const int stride, const int padding, const int dilation, const bool has_bias,
                const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits> bias) 
            {{
                const int batch_size = input.size(0);
                const int input_length = input.size(2);
                const int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + 2 * padding;
                
                const int b = blockIdx.x;
                const int oc = blockIdx.y;
                const int pos = threadIdx.x + blockIdx.z * blockDim.x;

                if (pos >= output_length) return;

                scalar_t sum = has_bias ? bias[oc] : 0;

                for (int ic = 0; ic < in_channels; ++ic) {{
                    for (int k = 0; k < kernel_size; ++k) {{
                        const int dilated_k = k * dilation;
                        const int input_pos = (pos - dilated_k - padding) / stride;
                        if ((input_pos >= 0) && (input_pos < input_length) && ((pos - dilated_k - padding) % stride == 0)) {{
                            sum += input[b][ic][input_pos] * weight[oc][ic][k];
                        }}
                    }}
                }}
                output[b][oc][pos] = sum;
            }}

            torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                             int in_channels, int out_channels, int kernel_size, int stride,
                                             int padding, int dilation, bool has_bias) {{
                const int batch_size = input.size(0);
                const int input_length = input.size(2);
                const int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + 2 * padding;

                auto output = torch::empty({{batch_size, out_channels, output_length}}, 
                                          input.options());
                
                const int threads = 256;
                const dim3 blocks(batch_size, out_channels, 
                                 (output_length + threads - 1) / threads);

                auto input_acc = input.packed_accessor<float,3,torch::RestrictPtrTraits>();
                auto weight_acc = weight.packed_accessor<float,3,torch::RestrictPtrTraits>();
                auto output_acc = output.packed_accessor<float,3,torch::RestrictPtrTraits>();
                auto bias_acc = (has_bias) ? bias.packed_accessor<float,1,torch::RestrictPtrTraits>() : 
                                            torch::Tensor();
                
                conv_transpose_1d_kernel<float><<<blocks, threads, 0, 
                                                  at::cuda::getCurrentCUDAStream()>>>(
                    input_acc, weight_acc, output_acc, in_channels, out_channels, kernel_size,
                    stride, padding, dilation, has_bias, bias_acc);
                
                CUDA_CHECK(cudaGetLastError());
                return output;
            }}
            """,
            functions=["conv_transpose_cuda"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose_cuda.conv_transpose_cuda(
            x.cuda(), self.weight.cuda(), self.bias.cuda() if self.bias is not None else torch.Tensor(),
            self.in_channels, self.out_channels, self.kernel_size, self.stride,
            self.padding, self.dilation, self.bias is not None
        ).cuda()