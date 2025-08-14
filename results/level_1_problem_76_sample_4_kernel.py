import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None
        # Load the custom conv1d CUDA kernel
        self.custom_conv1d = load_inline(
            name="custom_conv1d",
            cuda_sources=f"""
                #include <torch/extension.h>
                #include <cuda_runtime.h>
                #include <ATen/cuda/CUDAContext.h>

                template <typename scalar_t>
                __global__ void conv1d_kernel(const scalar_t* __restrict__ input,
                                             const scalar_t* __restrict__ weight,
                                             scalar_t* output,
                                             int batch_size,
                                             int in_channels,
                                             int out_channels,
                                             int input_length,
                                             int kernel_size,
                                             int stride,
                                             int dilation,
                                             int output_length) {{
                    int batch_idx = blockIdx.x;
                    int out_channel = blockIdx.y;
                    int out_pos = threadIdx.x;

                    // Compute output index
                    if (out_pos >= output_length) return;

                    scalar_t sum = 0;
                    for (int in_channel = 0; in_channel < in_channels; ++in_channel) {{
                        for (int k = 0; k < kernel_size; ++k) {{
                            int input_pos = out_pos * stride + k * dilation;
                            if (input_pos < input_length) {{
                                sum += input[batch_idx * in_channels * input_length + in_channel * input_length + input_pos] *
                                       weight[out_channel * in_channels * kernel_size + in_channel * kernel_size + k];
                            }}
                        }}
                    }}
                    if (bias != nullptr) {{
                        sum += bias[out_channel];
                    }}
                    output[batch_idx * out_channels * output_length + out_channel * output_length + out_pos] = sum;
                }}

                at::Tensor custom_conv1d_cuda(at::Tensor input, at::Tensor weight, at::Tensor bias, int stride, int dilation, int kernel_size) {{
                    const int batch_size = input.size(0);
                    const int in_channels = input.size(1);
                    const int input_length = input.size(2);
                    const int out_channels = weight.size(0);
                    const int output_length = (input_length - dilation * (kernel_size - 1) - 1)/stride + 1;

                    at::Tensor output = at::empty({{batch_size, out_channels, output_length}}, input.options());

                    int threads = 256;
                    dim3 blocks(batch_size, out_channels, 1);
                    dim3 t_per_b(threads, 1, 1);

                    AT_CUDA_KERNEL_CHECK( conv1d_kernel<scalar_t>
                        << <blocks, threads, 0, at::cuda::getCurrentCUDAStream() >> > (
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            output.data<scalar_t>(),
                            batch_size,
                            in_channels,
                            out_channels,
                            input_length,
                            kernel_size,
                            stride,
                            dilation,
                            output_length
                        )
                    );
                    return output;
                }}

                """,
            functions=["custom_conv1d_cuda"],
            verbose=True
        )

    def forward(self, x):
        return self.custom_conv1d.custom_conv1d_cuda(
            x.cuda(), self.weight.cuda(), self.bias.cuda() if self.bias is not None else None, self.stride, self.dilation, self.kernel_size
        )